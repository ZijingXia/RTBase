#pragma once

#include "Core.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"
#include <cmath>

#pragma warning( disable : 4244)

class SceneBounds
{
public:
	Vec3 sceneCentre;
	float sceneRadius;
};

class Light
{
public:
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec3 normal(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec3 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}
	Colour evaluate(const Vec3& wi)
	{
		if (fabsf(Dot(wi, triangle->gNormal())) > 0.0f)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 1.0f / triangle->area;
	}
	bool isArea()
	{
		return true;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return triangle->gNormal();
	}
	float totalIntegratedPower()
	{
		return (triangle->area * emission.Lum());
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		return triangle->sample(sampler, pdf);
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Add code to sample a direction from the light
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wi);
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
	}
};

class BackgroundColour : public Light
{
public:
	Colour emission;
	BackgroundColour(Colour _emission)
	{
		emission = _emission;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}
	Colour evaluate(const Vec3& wi)
	{
		return emission;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return SamplingDistributions::uniformSpherePDF(wi);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		return emission.Lum() * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light
{
public:
	Texture* env;
	float cachedIntegratedPower;
	std::vector<float> distribution;
	std::vector<float> cdf;
	float totalWeight = 0.0f;
	EnvironmentMap(Texture* _env)
	{
		env = _env;
		cachedIntegratedPower = -1.0f;
		buildDistribution();
	}
	void buildDistribution()
	{
		const int texelCount = env->width * env->height;
		distribution.assign(texelCount, 0.0f);
		cdf.assign(texelCount, 0.0f);
		totalWeight = 0.0f;
		for (int y = 0; y < env->height; y++)
		{
			const float v = ((float)y + 0.5f) / (float)env->height;
			const float theta = v * M_PI;
			const float sinTheta = sinf(theta);
			for (int x = 0; x < env->width; x++)
			{
				const int idx = (y * env->width) + x;
				const float w = std::max(0.0f, env->texels[idx].Lum() * sinTheta);
				distribution[idx] = w;
				totalWeight += w;
				cdf[idx] = totalWeight;
			}
		}
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Assignment: Update this code to importance sampling lighting based on luminance of each pixel
		Vec3 wi = sampleDirectionFromLight(sampler, pdf);
		reflectedColour = evaluate(wi);
		return wi;
	}
	Colour evaluate(const Vec3& wi)
	{
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		u = u / (2.0f * M_PI);
		float clampedY = std::max(-1.0f, std::min(1.0f, wi.y));
		float v = acosf(clampedY) / M_PI;
		return env->sample(u, v);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Assignment: Update this code to return the correct PDF of luminance weighted importance sampling
		if (totalWeight <= 0.0f || env->width <= 0 || env->height <= 0)
		{
			return SamplingDistributions::uniformSpherePDF(wi);
		}
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		u = u / (2.0f * M_PI);
		float clampedY = std::max(-1.0f, std::min(1.0f, wi.y));
		float v = acosf(clampedY) / M_PI;
		int x = std::min(env->width - 1, std::max(0, (int)(u * env->width)));
		int y = std::min(env->height - 1, std::max(0, (int)(v * env->height)));
		const int idx = (y * env->width) + x;
		const float texelProb = distribution[idx] / totalWeight;
		const float sinTheta = sqrtf(std::max(0.0f, 1.0f - (clampedY * clampedY)));
		if (sinTheta <= 0.0f)
		{
			return 0.0f;
		}
		const float pdfUV = texelProb * (float)(env->width * env->height);
		return pdfUV / (2.0f * M_PI * M_PI * sinTheta);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		if (cachedIntegratedPower >= 0.0f)
		{
			return cachedIntegratedPower;
		}
		float total = 0.0f;
		for (int i = 0; i < env->height; i++)
		{
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++)
			{
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		cachedIntegratedPower = total * 4.0f * M_PI;
		if (std::isfinite(cachedIntegratedPower) == false || cachedIntegratedPower < 0.0f)
		{
			cachedIntegratedPower = 0.0f;
		}
		return cachedIntegratedPower;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		// Samples a point on the bounding sphere of the scene. Feel free to improve this.
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 1.0f / (4 * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Replace this tabulated sampling of environment maps
		if (totalWeight <= 0.0f || cdf.empty())
		{
			Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
			pdf = SamplingDistributions::uniformSpherePDF(wi);
			return wi;
		}
		const float target = sampler->next() * totalWeight;
		int idx = (int)(std::lower_bound(cdf.begin(), cdf.end(), target) - cdf.begin());
		if (idx < 0)
		{
			idx = 0;
		}
		if (idx >= (int)cdf.size())
		{
			idx = (int)cdf.size() - 1;
		}
		const int y = idx / env->width;
		const int x = idx % env->width;
		const float u = ((float)x + sampler->next()) / (float)env->width;
		const float v = ((float)y + sampler->next()) / (float)env->height;
		const float phi = u * 2.0f * M_PI;
		const float theta = v * M_PI;
		const float sinTheta = sinf(theta);
		const float cosTheta = cosf(theta);
		Vec3 wi(cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);
		pdf = PDF(ShadingData(), wi);
		return wi;
	}
};