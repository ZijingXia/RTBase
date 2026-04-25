#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include <algorithm>
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>
#include <atomic>
#include <vector>
#include <chrono>
#include <mutex>
#include <condition_variable>
#if __has_include(<OpenImageDenoise/oidn.hpp>)
#include <OpenImageDenoise/oidn.hpp>
#define RTBASE_HAS_OIDN 1
#else
#define RTBASE_HAS_OIDN 0
#endif

class RayTracer
{
public:
	bool enableEnvironmentLight = true;
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom* samplers;
	int numProcs;
	std::vector<std::thread> workers;
	std::mutex workerMutex;
	std::condition_variable workerCV;
	std::condition_variable workerDoneCV;
	std::atomic<unsigned int> nextTile = 0;
	unsigned int totalTiles = 0;
	std::atomic<int> activeWorkers = 0;
	bool stopWorkers = false;
	int submittedFrameID = 0;
	int denoiseIntervalSPP = 4;
	std::vector<float> beautyAOV;
	std::vector<float> normalAOV;
	std::vector<float> albedoAOV;
	std::vector<float> denoisedAOV;
	bool denoiserEnabled = true;
	bool denoiseFailed = false;
#if RTBASE_HAS_OIDN
	oidn::DeviceRef oidnDevice;
#endif

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);

		numProcs = 11;

		samplers = new MTRandom[numProcs];
		for (int i = 0; i < numProcs; ++i)
		{
			samplers[i] = MTRandom((unsigned int)(1337 + (i * 977)));
		}
		for (int i = 0; i < numProcs; ++i)
		{
			workers.emplace_back([this, i]() { workerLoop(i); });
		}
		const size_t pixelCount = (size_t)film->width * (size_t)film->height;
		beautyAOV.resize(pixelCount * 3, 0.0f);
		normalAOV.resize(pixelCount * 3, 0.0f);
		albedoAOV.resize(pixelCount * 3, 0.0f);
		denoisedAOV.resize(pixelCount * 3, 0.0f);
#if RTBASE_HAS_OIDN
		oidnDevice = oidn::newDevice();
		oidnDevice.commit();
#endif

#if RTBASE_HAS_OIDN
		std::cout << "OIDN Enabled" << std::endl;
#else
		std::cout << "OIDN NOT found" << std::endl;
#endif

		clear();
	}
	void clear()
	{
		film->clear();
		std::fill(beautyAOV.begin(), beautyAOV.end(), 0.0f);
		std::fill(normalAOV.begin(), normalAOV.end(), 0.0f);
		std::fill(albedoAOV.begin(), albedoAOV.end(), 0.0f);
		std::fill(denoisedAOV.begin(), denoisedAOV.end(), 0.0f);
		denoiseFailed = false;
	}
	~RayTracer()
	{
		{
			std::lock_guard<std::mutex> lock(workerMutex);
			stopWorkers = true;
			submittedFrameID++;
		}
		workerCV.notify_all();
		for (std::thread& worker : workers)
		{
			if (worker.joinable())
			{
				worker.join();
			}
		}
		delete[] samplers;
	}
	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		// Is surface is specular we cannot computing direct lighting
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		// Compute direct lighting here
		Colour Ld(0.0f, 0.0f, 0.0f);
		const int directSamples = 2;

		auto powerHeuristic = [](float pdfA, float pdfB) -> float
			{
				float a2 = pdfA * pdfA;
				float b2 = pdfB * pdfB;
				float d = a2 + b2;
				if (d <= 0.0f)
				{
					return 0.0f;
				}
				return a2 / d;
			};
		auto lightSelectionPMF = [&](Light* target) -> float
			{
				if (target == NULL || scene->lights.empty())
				{
					return 0.0f;
				}
				float totalPower = 0.0f;
				float targetPower = 0.0f;
				for (size_t l = 0; l < scene->lights.size(); ++l)
				{
					float p = std::max(0.0f, scene->lights[l]->totalIntegratedPower());
					totalPower += p;
					if (scene->lights[l] == target)
					{
						targetPower = p;
					}
				}
				if (totalPower <= 0.0f)
				{
					return 1.0f / (float)scene->lights.size();
				}
				return targetPower / totalPower;
			};

		for (int i = 0; i < directSamples; i++)
		{
			float lightPMF = 0.0f;
			Light* light = scene->sampleLight(sampler, lightPMF);
			if (light == NULL || lightPMF <= 0.0f)
			{
				continue;
			}
			Colour Le(0.0f, 0.0f, 0.0f);
			float lightPDF = 0.0f;
			Vec3 wi(0.0f, 1.0f, 0.0f);
			//float geometryTerm = 1.0f;
			float lightPdfW = 0.0f;
			if (light->isArea())
			{
				Vec3 lightPos = light->sample(shadingData, sampler, Le, lightPDF);
				Vec3 toLight = lightPos - shadingData.x;
				float distSq = Dot(toLight, toLight);
				if (distSq <= 0.0f)
				{
					continue;
				}
				float dist = sqrtf(distSq);
				wi = toLight / dist;
				float cosSurface = std::max(0.0f, Dot(shadingData.sNormal, wi));
				if (cosSurface <= 0.0f)
				{
					continue;
				}
				Vec3 ln = light->normal(shadingData, wi);
				float cosLight = fabsf(Dot(ln, -wi));
				if (cosLight <= 0.0f || lightPDF <= 0.0f)
				{
					continue;
				}
				if (scene->visible(shadingData.x, lightPos) == false)
				{
					continue;
				}
				//geometryTerm = cosLight / distSq;
				lightPdfW = lightPDF * (distSq / std::max(EPSILON, cosLight));
			}
			else
			{
				wi = light->sample(shadingData, sampler, Le, lightPDF);
				if (lightPDF <= 0.0f)
				{
					continue;
				}
				if (Dot(shadingData.sNormal, wi) <= 0.0f)
				{
					continue;
				}
				Ray shadowRay;
				shadowRay.init(shadingData.x + (wi * EPSILON), wi);
				IntersectionData occluder = scene->traverse(shadowRay);
				if (occluder.t < FLT_MAX)
				{
					continue;
				}
				lightPdfW = lightPDF;
			}
			float cosTheta = std::max(0.0f, Dot(shadingData.sNormal, wi));
			Colour f = shadingData.bsdf->evaluate(shadingData, wi);
			//Colour f(0.8f / 3.14159265f, 0.8f / 3.14159265f, 0.8f / 3.14159265f);
			float lightPathPdf = lightPdfW * lightPMF;
			if (lightPathPdf > 0.0f)
			{
				float bsdfPdf = shadingData.bsdf->PDF(shadingData, wi);
				float wLight = powerHeuristic(lightPathPdf, bsdfPdf);
				Ld = Ld + ((f * Le) * ((cosTheta * wLight) / lightPathPdf));
			}
			Colour bsdfReflected(0.0f, 0.0f, 0.0f);
			float bsdfPdf = 0.0f;
			Vec3 bsdfWi = shadingData.bsdf->sample(shadingData, sampler, bsdfReflected, bsdfPdf);
			if (bsdfPdf <= 0.0f)
			{
				continue;
			}
			float bsdfCos = std::max(0.0f, Dot(shadingData.sNormal, bsdfWi));
			if (bsdfCos <= 0.0f)
			{
				continue;
			}
			Ray bsdfRay;
			bsdfRay.init(shadingData.x + (bsdfWi * EPSILON), bsdfWi);
			IntersectionData hit = scene->traverse(bsdfRay);
			Colour bsdfLe(0.0f, 0.0f, 0.0f);
			Light* hitLight = NULL;
			if (hit.t == FLT_MAX)
			{
				if (enableEnvironmentLight == false)
				{
					continue;
				}

				bsdfLe = scene->background->evaluate(bsdfWi);
				hitLight = scene->background;
			}
			else
			{
				ShadingData hitShading = scene->calculateShadingData(hit, bsdfRay);
				if (hitShading.bsdf->isLight() == false)
				{
					continue;
				}
				bsdfLe = hitShading.bsdf->emit(hitShading, hitShading.wo);
				for (size_t l = 0; l < scene->lights.size(); ++l)
				{
					AreaLight* area = dynamic_cast<AreaLight*>(scene->lights[l]);
					if (area != NULL && area->triangle == &scene->triangles[hit.ID])
					{
						hitLight = scene->lights[l];
						break;
					}
				}
				if (hitLight == NULL)
				{
					continue;
				}
			}
			float hitLightPMF = lightSelectionPMF(hitLight);
			float hitLightPdfW = hitLight->PDF(shadingData, bsdfWi);
			if (hitLight->isArea())
			{
				Vec3 lightPoint = bsdfRay.at(hit.t);
				Vec3 toLight = lightPoint - shadingData.x;
				float distSq = Dot(toLight, toLight);
				Vec3 ln = hitLight->normal(shadingData, bsdfWi);
				float cosLight = fabsf(Dot(ln, -bsdfWi));
				if (cosLight <= 0.0f)
				{
					continue;
				}
				hitLightPdfW = hitLightPdfW * (distSq / std::max(EPSILON, cosLight));
			}
			float hitLightPathPdf = hitLightPMF * hitLightPdfW;
			float wBSDF = powerHeuristic(bsdfPdf, hitLightPathPdf);
			Colour bsdfF = shadingData.bsdf->evaluate(shadingData, bsdfWi);
			Ld = Ld + ((bsdfF * bsdfLe) * ((bsdfCos * wBSDF) / bsdfPdf));
		}
		return Ld / (float)directSamples;
	}
	Colour pathTrace(Ray& r, const Colour& pathThroughput, int depth, Sampler* sampler, Colour* normalOut = nullptr, Colour* albedoOut = nullptr)
	{
		// Add pathtracer code here
		if (depth > 5)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t == FLT_MAX)
		{
			if (depth == 0)
			{
				if (normalOut != nullptr)
				{
					*normalOut = Colour(0.0f, 0.0f, 0.0f);
				}
				if (albedoOut != nullptr)
				{
					if (enableEnvironmentLight)
					{
						*albedoOut = Colour(0.0f, 0.0f, 0.0f);
					}
					else
					{
						*albedoOut = Colour(0.0f, 0.0f, 0.0f);
					}
				}
			}

			if (enableEnvironmentLight)
			{
				return pathThroughput * scene->background->evaluate(r.dir);
			}

			return Colour(0.0f, 0.0f, 0.0f);
		}
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (depth == 0)
		{
			if (normalOut != nullptr)
			{
				*normalOut = Colour(shadingData.sNormal.x, shadingData.sNormal.y, shadingData.sNormal.z);
			}
			if (albedoOut != nullptr)
			{
				if (shadingData.bsdf->isLight())
				{
					*albedoOut = Colour(0.0f, 0.0f, 0.0f);
				}
				else
				{
					*albedoOut = shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
				}
			}
		}
		if (shadingData.bsdf->isLight())
		{
			return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
		}
		Colour L(0.0f, 0.0f, 0.0f);
		if (shadingData.bsdf->isPureSpecular() == false)
		{
			L = pathThroughput * computeDirect(shadingData, sampler);
		}
		Colour reflectedColour(0.0f, 0.0f, 0.0f);
		float pdf = 0.0f;
		Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, reflectedColour, pdf);
		if (pdf <= 0.0f)
		{
			return L;
		}
		float cosTheta = fabsf(Dot(shadingData.sNormal, wi));
		Colour newThroughput = pathThroughput * reflectedColour * (cosTheta / pdf);
		if (depth > 2)
		{
			float rr = std::min(0.95f, std::max(newThroughput.r, std::max(newThroughput.g, newThroughput.b)));
			if (sampler->next() > rr)
			{
				return L;
			}
			newThroughput = newThroughput / rr;
		}
		Ray nextRay;
		nextRay.init(shadingData.x + (wi * EPSILON), wi);
		return L + pathTrace(nextRay, newThroughput, depth + 1, sampler, normalOut, albedoOut);
	}
	Colour direct(Ray& r, Sampler* sampler)
	{
		// Compute direct lighting for an image sampler here
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t == FLT_MAX)
		{
			return scene->background->evaluate(r.dir);
		}
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.bsdf->isLight())
		{
			return shadingData.bsdf->emit(shadingData, shadingData.wo);
		}
		return computeDirect(shadingData, sampler);
	}
	Colour albedo(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate(r.dir);
	}
	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	void render()
	{
		film->incrementSPP();
		const unsigned int tileSize = 16;
		const unsigned int tilesX = (film->width + tileSize - 1) / tileSize;
		const unsigned int tilesY = (film->height + tileSize - 1) / tileSize;
		totalTiles = tilesX * tilesY;
		nextTile.store(0);
		activeWorkers.store(numProcs);
		{
			std::lock_guard<std::mutex> lock(workerMutex);
			submittedFrameID++;
		}
		workerCV.notify_all();
		{
			std::unique_lock<std::mutex> lock(workerMutex);
			workerDoneCV.wait(lock, [this]() { return activeWorkers.load() == 0; });
		}
		if (denoiserEnabled && denoiseFailed == false && (film->SPP % denoiseIntervalSPP == 0))
		{
			std::cout << "OIDN Running at SPP = " << film->SPP << std::endl;
			runDenoiser();
		}
		else
		{
			for (unsigned int y = 0; y < film->height; y++)
			{
				for (unsigned int x = 0; x < film->width; x++)
				{
					unsigned char r;
					unsigned char g;
					unsigned char b;
					film->tonemap(x, y, r, g, b);
					canvas->draw(x, y, r, g, b);
				}
			}

			/*for (unsigned int y = 0; y < film->height; y++)
			{
				for (unsigned int x = 0; x < film->width; x++)
				{
					const unsigned int idx = ((y * film->width) + x) * 3;

					float invSPP = 1.0f / (float)film->SPP;

					float r = albedoAOV[idx] * invSPP;
					float g = albedoAOV[idx + 1] * invSPP;
					float b = albedoAOV[idx + 2] * invSPP;

					unsigned char rc = (unsigned char)(std::min(1.0f, r) * 255.0f);
					unsigned char gc = (unsigned char)(std::min(1.0f, g) * 255.0f);
					unsigned char bc = (unsigned char)(std::min(1.0f, b) * 255.0f);

					canvas->draw(x, y, rc, gc, bc);
				}
			}*/

			/*for (unsigned int y = 0; y < film->height; y++)
			{
				for (unsigned int x = 0; x < film->width; x++)
				{
					const unsigned int idx = ((y * film->width) + x) * 3;

					float invSPP = 1.0f / (float)film->SPP;

					float r = normalAOV[idx] * invSPP;
					float g = normalAOV[idx + 1] * invSPP;
					float b = normalAOV[idx + 2] * invSPP;

					unsigned char rc = (unsigned char)(std::min(1.0f, r) * 255.0f);
					unsigned char gc = (unsigned char)(std::min(1.0f, g) * 255.0f);
					unsigned char bc = (unsigned char)(std::min(1.0f, b) * 255.0f);

					canvas->draw(x, y, rc, gc, bc);
				}
			}*/

		}

	}
	int getSPP()
	{
		return film->SPP;
	}
	void saveHDR(std::string filename)
	{
		film->save(filename);
	}
	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}

private:
	void workerLoop(int threadID)
	{
		int observedFrameID = 0;
		const bool skipSplatForBoxFilter = (dynamic_cast<BoxFilter*>(film->filter) != NULL);
		while (true)
		{
			{
				std::unique_lock<std::mutex> lock(workerMutex);
				workerCV.wait(lock, [this, &observedFrameID]() { return stopWorkers || submittedFrameID > observedFrameID; });
				if (stopWorkers)
				{
					return;
				}
				observedFrameID = submittedFrameID;
			}
			while (true)
			{
				unsigned int tileID = nextTile.fetch_add(1);
				if (tileID >= totalTiles)
				{
					break;
				}
				shadeTile(tileID, threadID, skipSplatForBoxFilter);
			}
			if (activeWorkers.fetch_sub(1) == 1)
			{
				std::lock_guard<std::mutex> lock(workerMutex);
				workerDoneCV.notify_one();
			}
		}
	}
	void shadeTile(unsigned int tileID, int threadID, bool skipSplatForBoxFilter)
	{
		const unsigned int tileSize = 16;
		const unsigned int tilesX = (film->width + tileSize - 1) / tileSize;
		unsigned int tileX = tileID % tilesX;
		unsigned int tileY = tileID / tilesX;
		unsigned int x0 = tileX * tileSize;
		unsigned int y0 = tileY * tileSize;
		unsigned int x1 = std::min(x0 + tileSize, film->width);
		unsigned int y1 = std::min(y0 + tileSize, film->height);
		for (unsigned int y = y0; y < y1; y++)
		{
			for (unsigned int x = x0; x < x1; x++)
			{
				float px = x + samplers[threadID].next();
				float py = y + samplers[threadID].next();
				Ray ray = scene->camera.generateRay(px, py);

				Colour normal(0.0f, 0.0f, 0.0f);
				Colour alb(0.0f, 0.0f, 0.0f);
				Colour col = pathTrace(ray, Colour(1.0f, 1.0f, 1.0f), 0, &samplers[threadID], &normal, &alb);
				const unsigned int idx = ((y * film->width) + x) * 3;
				beautyAOV[idx] += col.r;
				beautyAOV[idx + 1] += col.g;
				beautyAOV[idx + 2] += col.b;
				normalAOV[idx] += normal.r;
				normalAOV[idx + 1] += normal.g;
				normalAOV[idx + 2] += normal.b;
				albedoAOV[idx] += alb.r;
				albedoAOV[idx + 1] += alb.g;
				albedoAOV[idx + 2] += alb.b;

				if (skipSplatForBoxFilter)
				{
					const unsigned int pixelID = (y * film->width) + x;
					film->film[pixelID] = film->film[pixelID] + col;
				}
				else
				{
					film->splat(px, py, col);
				}
			}
		}
	}
	void runDenoiser()
	{
#if RTBASE_HAS_OIDN
		if (film->SPP <= 0)
		{
			return;
		}

		const float invSPP = 1.0f / (float)film->SPP;

		std::vector<float> beautyInput = beautyAOV;
		std::vector<float> normalInput = normalAOV;
		std::vector<float> albedoInput = albedoAOV;

		for (size_t i = 0; i < beautyInput.size(); ++i)
		{
			beautyInput[i] *= invSPP;
			normalInput[i] *= invSPP;
			albedoInput[i] *= invSPP;

			if (!std::isfinite(beautyInput[i])) beautyInput[i] = 0.0f;
			if (!std::isfinite(normalInput[i])) normalInput[i] = 0.0f;
			if (!std::isfinite(albedoInput[i])) albedoInput[i] = 0.0f;

			beautyInput[i] = std::max(0.0f, beautyInput[i]);
			albedoInput[i] = std::max(0.0f, albedoInput[i]);
		}

		const size_t bytes = beautyInput.size() * sizeof(float);

		oidn::BufferRef colorBuf = oidnDevice.newBuffer(bytes);
		oidn::BufferRef normalBuf = oidnDevice.newBuffer(bytes);
		oidn::BufferRef albedoBuf = oidnDevice.newBuffer(bytes);
		oidn::BufferRef outputBuf = oidnDevice.newBuffer(bytes);

		colorBuf.write(0, bytes, beautyInput.data());
		normalBuf.write(0, bytes, normalInput.data());
		albedoBuf.write(0, bytes, albedoInput.data());

		oidn::FilterRef filter = oidnDevice.newFilter("RT");

		filter.setImage("color", colorBuf, oidn::Format::Float3, film->width, film->height);
		filter.setImage("normal", normalBuf, oidn::Format::Float3, film->width, film->height);
		filter.setImage("albedo", albedoBuf, oidn::Format::Float3, film->width, film->height);
		filter.setImage("output", outputBuf, oidn::Format::Float3, film->width, film->height);

		filter.set("hdr", true);
		filter.commit();
		filter.execute();

		const char* errorMessage = nullptr;
		if (oidnDevice.getError(errorMessage) != oidn::Error::None)
		{
			std::cout << "[OIDN] Error: " << errorMessage << std::endl;
			denoiseFailed = true;
			return;
		}

		outputBuf.read(0, bytes, denoisedAOV.data());

		for (unsigned int y = 0; y < film->height; ++y)
		{
			for (unsigned int x = 0; x < film->width; ++x)
			{
				const unsigned int idx = ((y * film->width) + x) * 3;

				Colour c(
					denoisedAOV[idx],
					denoisedAOV[idx + 1],
					denoisedAOV[idx + 2]
				);

				if (!std::isfinite(c.r)) c.r = 0.0f;
				if (!std::isfinite(c.g)) c.g = 0.0f;
				if (!std::isfinite(c.b)) c.b = 0.0f;

				c.r = std::max(0.0f, c.r);
				c.g = std::max(0.0f, c.g);
				c.b = std::max(0.0f, c.b);

				c.r = c.r / (1.0f + c.r);
				c.g = c.g / (1.0f + c.g);
				c.b = c.b / (1.0f + c.b);

				float invGamma = 1.0f / 2.2f;
				c.r = powf(c.r, invGamma);
				c.g = powf(c.g, invGamma);
				c.b = powf(c.b, invGamma);

				unsigned char r = (unsigned char)(std::min(1.0f, c.r) * 255.0f);
				unsigned char g = (unsigned char)(std::min(1.0f, c.g) * 255.0f);
				unsigned char b = (unsigned char)(std::min(1.0f, c.b) * 255.0f);

				canvas->draw(x, y, r, g, b);
			}
		}
#else
		denoiseFailed = true;
#endif
	}
};