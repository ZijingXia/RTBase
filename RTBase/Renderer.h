#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>
#include <atomic>

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;
	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];
		clear();
	}
	void clear()
	{
		film->clear();
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
		const int directSamples = 4;
		for (int i = 0; i < directSamples; i++)
		{
			Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
			float pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);
			if (pdf <= 0.0f)
			{
				continue;
			}
			Vec3 wi = shadingData.frame.toWorld(wiLocal);
			float cosTheta = std::max(0.0f, Dot(shadingData.sNormal, wi));
			if (cosTheta <= 0.0f)
			{
				continue;
			}
			Ray shadowRay;
			shadowRay.init(shadingData.x + (wi * EPSILON), wi);
			IntersectionData hit = scene->traverse(shadowRay);
			Colour Li = scene->background->evaluate(wi);
			if (hit.t < FLT_MAX)
			{
				ShadingData lightSD = scene->calculateShadingData(hit, shadowRay);
				if (lightSD.bsdf != NULL && lightSD.bsdf->isLight())
				{
					Li = lightSD.bsdf->emit(lightSD, -wi);
				}
				else
				{
					Li = Colour(0.0f, 0.0f, 0.0f);
				}
			}
			Colour f = shadingData.bsdf->evaluate(shadingData, wi);
			Ld = Ld + ((f * Li) * (cosTheta / pdf));
		}
		return Ld / (float)directSamples;
	}
	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler)
	{
		// Add pathtracer code here
		if (depth > 5)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t == FLT_MAX)
		{
			return pathThroughput * scene->background->evaluate(r.dir);
		}
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.bsdf->isLight())
		{
			return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
		}
		Colour L = pathThroughput * computeDirect(shadingData, sampler);
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
		return L + pathTrace(nextRay, newThroughput, depth + 1, sampler);
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
		const unsigned int totalTiles = tilesX * tilesY;
		std::atomic<unsigned int> nextTile(0);
		auto worker = [&](int threadID)
			{
				while (true)
				{
					unsigned int tileID = nextTile.fetch_add(1);
					if (tileID >= totalTiles)
					{
						break;
					}
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
							float px = x + 0.5f;
							float py = y + 0.5f;
							Ray ray = scene->camera.generateRay(px, py);
							
							Colour col = viewNormals(ray);
							
							film->splat(px, py, col);
							unsigned char r = (unsigned char)(col.r * 255);
							unsigned char g = (unsigned char)(col.g * 255);
							unsigned char b = (unsigned char)(col.b * 255);
							film->tonemap(x, y, r, g, b);

							canvas->draw(x, y, r, g, b);
						}
					}
				}
			};
		for (int i = 0; i < numProcs; i++)
		{
			threads[i] = new std::thread(worker, i);
		}
		for (int i = 0; i < numProcs; i++)
		{
			threads[i]->join();
			delete threads[i];
			threads[i] = NULL;
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
};