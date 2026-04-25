#pragma once

#include "Core.h"
#include "Imaging.h"
#include "Sampling.h"

#pragma warning( disable : 4244)
#pragma warning( disable : 4305) // Double to float

class BSDF;

class ShadingData
{
public:
	Vec3 x;
	Vec3 wo;
	Vec3 sNormal;
	Vec3 gNormal;
	float tu;
	float tv;
	Frame frame;
	BSDF* bsdf;
	float t;
	ShadingData() {}
	ShadingData(Vec3 _x, Vec3 n)
	{
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = NULL;
	}
};

class ShadingHelper
{
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt)
	{
		// Add code here
		cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));
		float etaI = iorExt;
		float etaT = iorInt;
		if (cosTheta <= 0.0f)
		{
			std::swap(etaI, etaT);
			cosTheta = fabsf(cosTheta);
		}
		float sinThetaI = sqrtf(std::max(0.0f, 1.0f - (cosTheta * cosTheta)));
		float sinThetaT = (etaI / etaT) * sinThetaI;
		if (sinThetaT >= 1.0f)
		{
			return 1.0f;
		}
		float cosThetaT = sqrtf(std::max(0.0f, 1.0f - (sinThetaT * sinThetaT)));
		float rs = ((etaT * cosTheta) - (etaI * cosThetaT)) / ((etaT * cosTheta) + (etaI * cosThetaT));
		float rp = ((etaI * cosTheta) - (etaT * cosThetaT)) / ((etaI * cosTheta) + (etaT * cosThetaT));
		return 0.5f * ((rs * rs) + (rp * rp));
	}
	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k)
	{
		// Add code here
		cosTheta = std::max(0.0f, std::min(1.0f, cosTheta));
		float cosThetaSq = cosTheta * cosTheta;
		float sinThetaSq = std::max(0.0f, 1.0f - cosThetaSq);

		Colour etaSq = ior * ior;
		Colour kSq = k * k;
		Colour t0 = etaSq - kSq - Colour(sinThetaSq, sinThetaSq, sinThetaSq);
		Colour a2pb2(
			sqrtf(std::max(0.0f, (t0.r * t0.r) + (4.0f * etaSq.r * kSq.r))),
			sqrtf(std::max(0.0f, (t0.g * t0.g) + (4.0f * etaSq.g * kSq.g))),
			sqrtf(std::max(0.0f, (t0.b * t0.b) + (4.0f * etaSq.b * kSq.b)))
		);

		Colour a(
			sqrtf(std::max(0.0f, 0.5f * (a2pb2.r + t0.r))),
			sqrtf(std::max(0.0f, 0.5f * (a2pb2.g + t0.g))),
			sqrtf(std::max(0.0f, 0.5f * (a2pb2.b + t0.b)))
		);

		Colour rsNum = Colour((a2pb2.r + cosThetaSq) - (2.0f * a.r * cosTheta),
			(a2pb2.g + cosThetaSq) - (2.0f * a.g * cosTheta),
			(a2pb2.b + cosThetaSq) - (2.0f * a.b * cosTheta));
		Colour rsDen = Colour((a2pb2.r + cosThetaSq) + (2.0f * a.r * cosTheta),
			(a2pb2.g + cosThetaSq) + (2.0f * a.g * cosTheta),
			(a2pb2.b + cosThetaSq) + (2.0f * a.b * cosTheta));
		Colour Rs = rsNum / rsDen;

		Colour t1 = a2pb2 * cosThetaSq;
		Colour t2 = Colour(sinThetaSq * sinThetaSq, sinThetaSq * sinThetaSq, sinThetaSq * sinThetaSq);
		Colour rpNum = Rs * (t1 + t2 - Colour(2.0f * a.r * cosTheta * sinThetaSq, 2.0f * a.g * cosTheta * sinThetaSq, 2.0f * a.b * cosTheta * sinThetaSq));
		Colour rpDen = t1 + t2 + Colour(2.0f * a.r * cosTheta * sinThetaSq, 2.0f * a.g * cosTheta * sinThetaSq, 2.0f * a.b * cosTheta * sinThetaSq);
		Colour Rp = rpNum / rpDen;

		return (Rs + Rp) * 0.5f;
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		// Add code here
		float absCosTheta = fabsf(wi.z);
		if (absCosTheta <= 0.0f)
		{
			return 0.0f;
		}
		float sinThetaSq = std::max(0.0f, 1.0f - (absCosTheta * absCosTheta));
		float tanThetaSq = sinThetaSq / std::max(EPSILON, absCosTheta * absCosTheta);
		float alphaSq = alpha * alpha;
		return 0.5f * (-1.0f + sqrtf(1.0f + (alphaSq * tanThetaSq)));
	}
	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		// Add code here
		return 1.0f / (1.0f + lambdaGGX(wi, alpha) + lambdaGGX(wo, alpha));
	}
	static float Dggx(Vec3 h, float alpha)
	{
		// Add code here
		float cosThetaH = std::max(0.0f, h.z);
		if (cosThetaH <= 0.0f)
		{
			return 0.0f;
		}
		float alphaSq = alpha * alpha;
		float denom = (cosThetaH * cosThetaH) * (alphaSq - 1.0f) + 1.0f;
		return alphaSq / (M_PI * denom * denom);
	}
};

class BSDF
{
public:
	Colour emission;
	BSDF()
	{
		emission = Colour(0.0f, 0.0f, 0.0f);
	}
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight()
	{
		return emission.Lum() > 0 ? true : false;
	}
	void addLight(Colour _emission)
	{
		emission = _emission;
	}
	Colour emit(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	virtual float mask(const ShadingData& shadingData) = 0;
};


class DiffuseBSDF : public BSDF
{
public:
	Texture* albedo;
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add correct sampling code here
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wi);
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add correct PDF code here
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class MirrorBSDF : public BSDF
{
public:
	Texture* albedo;
	MirrorBSDF() = default;
	MirrorBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Mirror sampling code ()
		Vec3 wi = (shadingData.sNormal * (2.0f * Dot(shadingData.wo, shadingData.sNormal))) - shadingData.wo;
		wi = wi.normalize();
		pdf = 1.0f;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / std::max(EPSILON, fabsf(Dot(shadingData.sNormal, wi)));
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Mirror evaluation code ()
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Mirror PDF ()
		return 0.0f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};


class ConductorBSDF : public BSDF
{
public:
	Texture* albedo;
	Colour eta;
	Colour k;
	float alpha;
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
	{
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Conductor sampling code (done)
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		float r1 = sampler->next();
		float r2 = sampler->next();
		float phi = 2.0f * M_PI * r1;
		float tanThetaSq = (alpha * alpha) * r2 / std::max(EPSILON, 1.0f - r2);
		float cosTheta = 1.0f / sqrtf(1.0f + tanThetaSq);
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - (cosTheta * cosTheta)));
		Vec3 hLocal(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
		if (Dot(woLocal, hLocal) < 0.0f)
		{
			hLocal = -hLocal;
		}
		Vec3 wiLocal = ((hLocal * (2.0f * Dot(woLocal, hLocal))) - woLocal).normalize();
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return shadingData.frame.toWorld(Vec3(0.0f, 0.0f, 1.0f));
		}
		float D = ShadingHelper::Dggx(hLocal, alpha);
		float pdfH = D * std::max(0.0f, hLocal.z);
		pdf = pdfH / std::max(EPSILON, 4.0f * fabsf(Dot(woLocal, hLocal)));
		Vec3 wi = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Conductor evaluation code (done)
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		float cosI = wiLocal.z;
		float cosO = woLocal.z;
		if (cosI <= 0.0f || cosO <= 0.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		Vec3 h = (wiLocal + woLocal).normalize();
		if (h.z <= 0.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		float D = ShadingHelper::Dggx(h, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		Colour F = ShadingHelper::fresnelConductor(std::max(0.0f, Dot(wiLocal, h)), eta, k);
		Colour base = albedo->sample(shadingData.tu, shadingData.tv);
		return base * F * (D * G / std::max(EPSILON, 4.0f * cosI * cosO));
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Conductor PDF (done)
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
		{
			return 0.0f;
		}
		Vec3 h = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(h, alpha);
		float pdfH = D * std::max(0.0f, h.z);
		return pdfH / std::max(EPSILON, 4.0f * fabsf(Dot(woLocal, h)));
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class GlassBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	GlassBSDF() = default;
	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Glass sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (fabsf(woLocal.z) <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return shadingData.frame.toWorld(Vec3(0.0f, 0.0f, 1.0f));
		}

		bool entering = woLocal.z > 0.0f;
		float etaI = entering ? extIOR : intIOR;
		float etaT = entering ? intIOR : extIOR;
		float eta = etaI / etaT;
		float cosThetaO = woLocal.z;
		float cosThetaOAbs = fabsf(woLocal.z);
		float F = ShadingHelper::fresnelDielectric(cosThetaO, intIOR, extIOR);

		Vec3 wiLocal;
		if (sampler->next() < F)
		{
			wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z).normalize();
			pdf = F;
		}
		else
		{
			float sinThetaOSq = std::max(0.0f, 1.0f - (cosThetaOAbs * cosThetaOAbs));
			float sinThetaISq = eta * eta * sinThetaOSq;
			if (sinThetaISq >= 1.0f)
			{
				wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z).normalize();
				pdf = 1.0f;
				reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / std::max(EPSILON, fabsf(wiLocal.z));
				return shadingData.frame.toWorld(wiLocal);
			}
			float cosThetaI = sqrtf(std::max(0.0f, 1.0f - sinThetaISq));
			float sign = entering ? -1.0f : 1.0f;
			wiLocal = Vec3(-eta * woLocal.x, -eta * woLocal.y, sign * cosThetaI).normalize();
			pdf = std::max(0.0f, 1.0f - F);
		}
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / std::max(EPSILON, fabsf(wiLocal.z));
		return shadingData.frame.toWorld(wiLocal);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Glass evaluation code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (fabsf(woLocal.z) <= 0.0f || fabsf(wiLocal.z) <= 0.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		bool entering = woLocal.z > 0.0f;
		float etaI = entering ? extIOR : intIOR;
		float etaT = entering ? intIOR : extIOR;
		float eta = etaI / etaT;
		float cosThetaO = woLocal.z;
		float cosThetaOAbs = fabsf(woLocal.z);
		float F = ShadingHelper::fresnelDielectric(cosThetaO, intIOR, extIOR);
		Vec3 wrLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z).normalize();
		const float dirEps = 1.0e-4f;
		if (Dot(wiLocal, wrLocal) > (1.0f - dirEps))
		{
			return (albedo->sample(shadingData.tu, shadingData.tv) * F) / std::max(EPSILON, fabsf(wiLocal.z));
		}
		float sinThetaOSq = std::max(0.0f, 1.0f - (cosThetaOAbs * cosThetaOAbs));
		float sinThetaISq = eta * eta * sinThetaOSq;
		if (sinThetaISq >= 1.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		float cosThetaI = sqrtf(std::max(0.0f, 1.0f - sinThetaISq));
		float sign = entering ? -1.0f : 1.0f;
		Vec3 wtLocal = Vec3(-eta * woLocal.x, -eta * woLocal.y, sign * cosThetaI).normalize();
		if (Dot(wiLocal, wtLocal) > (1.0f - dirEps))
		{
			return (albedo->sample(shadingData.tu, shadingData.tv) * (1.0f - F)) / std::max(EPSILON, fabsf(wiLocal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with GlassPDF
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (fabsf(woLocal.z) <= 0.0f || fabsf(wiLocal.z) <= 0.0f)
		{
			return 0.0f;
		}
		bool entering = woLocal.z > 0.0f;
		float etaI = entering ? extIOR : intIOR;
		float etaT = entering ? intIOR : extIOR;
		float eta = etaI / etaT;
		float cosThetaO = woLocal.z;
		float cosThetaOAbs = fabsf(woLocal.z);
		float F = ShadingHelper::fresnelDielectric(cosThetaO, intIOR, extIOR);
		Vec3 wrLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z).normalize();
		const float dirEps = 1.0e-4f;
		if (Dot(wiLocal, wrLocal) > (1.0f - dirEps))
		{
			return F;
		}
		float sinThetaOSq = std::max(0.0f, 1.0f - (cosThetaOAbs * cosThetaOAbs));
		float sinThetaISq = eta * eta * sinThetaOSq;
		if (sinThetaISq >= 1.0f)
		{
			return 0.0f;
		}
		float cosThetaI = sqrtf(std::max(0.0f, 1.0f - sinThetaISq));
		float sign = entering ? -1.0f : 1.0f;
		Vec3 wtLocal = Vec3(-eta * woLocal.x, -eta * woLocal.y, sign * cosThetaI).normalize();
		if (Dot(wiLocal, wtLocal) > (1.0f - dirEps))
		{
			return std::max(0.0f, 1.0f - F);
		}
		return 0.0f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class GGXMicrofacetBSDF : public BSDF
{
public:
	Texture* albedo;
	float alpha;
	GGXMicrofacetBSDF() = default;
	GGXMicrofacetBSDF(Texture* _albedo, float roughness)
	{
		albedo = _albedo;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return shadingData.frame.toWorld(Vec3(0.0f, 0.0f, 1.0f));
		}
		float u1 = sampler->next();
		float u2 = sampler->next();
		float phi = 2.0f * M_PI * u1;
		float tanThetaSq = (alpha * alpha) * u2 / std::max(EPSILON, 1.0f - u2);
		float cosTheta = 1.0f / sqrtf(1.0f + tanThetaSq);
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - (cosTheta * cosTheta)));
		Vec3 hLocal(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
		if (Dot(woLocal, hLocal) <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return shadingData.frame.toWorld(Vec3(0.0f, 0.0f, 1.0f));
		}
		Vec3 wiLocal = ((hLocal * (2.0f * Dot(woLocal, hLocal))) - woLocal).normalize();
		if (wiLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return shadingData.frame.toWorld(Vec3(0.0f, 0.0f, 1.0f));
		}
		float D = ShadingHelper::Dggx(hLocal, alpha);
		float pdfH = D * std::max(0.0f, hLocal.z);
		pdf = pdfH / std::max(EPSILON, 4.0f * fabsf(Dot(woLocal, hLocal)));
		Vec3 wi = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		Vec3 h = (wiLocal + woLocal).normalize();
		if (h.z <= 0.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		float D = ShadingHelper::Dggx(h, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		Colour F = albedo->sample(shadingData.tu, shadingData.tv);
		return F * (D * G / std::max(EPSILON, 4.0f * wiLocal.z * woLocal.z));
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
		{
			return 0.0f;
		}
		Vec3 h = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(h, alpha);
		float pdfH = D * std::max(0.0f, h.z);
		return pdfH / std::max(EPSILON, 4.0f * fabsf(Dot(woLocal, h)));
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Dielectric sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo;
	float sigma;
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with OrenNayar sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		wi = shadingData.frame.toWorld(wi);
		reflectedColour = evaluate(shadingData, wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar evaluation code
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		float sigmaSq = sigma * sigma;
		float A = 1.0f - (sigmaSq / (2.0f * (sigmaSq + 0.33f)));
		float B = 0.45f * sigmaSq / (sigmaSq + 0.09f);

		float sinThetaI = sqrtf(std::max(0.0f, 1.0f - (wiLocal.z * wiLocal.z)));
		float sinThetaO = sqrtf(std::max(0.0f, 1.0f - (woLocal.z * woLocal.z)));
		float maxCos = 0.0f;
		if (sinThetaI > EPSILON && sinThetaO > EPSILON)
		{
			float sinPhiI = wiLocal.y / sinThetaI;
			float cosPhiI = wiLocal.x / sinThetaI;
			float sinPhiO = woLocal.y / sinThetaO;
			float cosPhiO = woLocal.x / sinThetaO;
			maxCos = std::max(0.0f, (cosPhiI * cosPhiO) + (sinPhiI * sinPhiO));
		}

		float sinAlpha;
		float tanBeta;
		if (fabsf(wiLocal.z) > fabsf(woLocal.z))
		{
			sinAlpha = sinThetaO;
			tanBeta = sinThetaI / std::max(EPSILON, fabsf(wiLocal.z));
		}
		else
		{
			sinAlpha = sinThetaI;
			tanBeta = sinThetaO / std::max(EPSILON, fabsf(woLocal.z));
		}
		float orenNayar = A + (B * maxCos * sinAlpha * tanBeta);
		return (albedo->sample(shadingData.tu, shadingData.tv) / M_PI) * orenNayar;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	float alphaToPhongExponent()
	{
		return (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Plastic sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		float F = ShadingHelper::fresnelDielectric(std::max(0.0f, woLocal.z), intIOR, extIOR);
		float specProb = std::max(0.05f, std::min(0.95f, F));
		Vec3 wiLocal;
		if (sampler->next() < specProb)
		{
			float exponent = alphaToPhongExponent();
			float u1 = sampler->next();
			float u2 = sampler->next();
			float cosTheta = powf(u1, 1.0f / (exponent + 1.0f));
			float sinTheta = sqrtf(std::max(0.0f, 1.0f - (cosTheta * cosTheta)));
			float phi = 2.0f * M_PI * u2;
			Vec3 localSpec(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

			Vec3 r = Vec3(-woLocal.x, -woLocal.y, woLocal.z).normalize();
			Frame specFrame;
			specFrame.fromVector(r);
			wiLocal = specFrame.toWorld(localSpec).normalize();
		}
		else
		{
			wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		}
		if (wiLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return shadingData.frame.toWorld(Vec3(0.0f, 0.0f, 1.0f));
		}
		Vec3 wi = shadingData.frame.toWorld(wiLocal);
		pdf = PDF(shadingData, wi);
		reflectedColour = evaluate(shadingData, wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic evaluation code
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		float F = ShadingHelper::fresnelDielectric(std::max(0.0f, woLocal.z), intIOR, extIOR);
		Colour kd = albedo->sample(shadingData.tu, shadingData.tv) * (1.0f - F);
		Colour diffuseTerm = kd / M_PI;

		Vec3 r = Vec3(-woLocal.x, -woLocal.y, woLocal.z).normalize();
		float cosAlpha = std::max(0.0f, Dot(r, wiLocal));
		float exponent = alphaToPhongExponent();
		float phongNorm = (exponent + 2.0f) / (2.0f * M_PI);
		Colour specularTerm = Colour(F, F, F) * (phongNorm * powf(cosAlpha, exponent));
		return diffuseTerm + specularTerm;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
		{
			return 0.0f;
		}
		float F = ShadingHelper::fresnelDielectric(std::max(0.0f, woLocal.z), intIOR, extIOR);
		float specProb = std::max(0.05f, std::min(0.95f, F));
		float diffusePDF = SamplingDistributions::cosineHemispherePDF(wiLocal);
		Vec3 r = Vec3(-woLocal.x, -woLocal.y, woLocal.z).normalize();
		float cosAlpha = std::max(0.0f, Dot(r, wiLocal));
		float exponent = alphaToPhongExponent();
		float specPDF = ((exponent + 1.0f) / (2.0f * M_PI)) * powf(cosAlpha, exponent);
		return ((1.0f - specProb) * diffusePDF) + (specProb * specPDF);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class LayeredBSDF : public BSDF
{
public:
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR;
	float extIOR;
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR)
	{
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}
	bool isPureSpecular()
	{
		return base->isPureSpecular();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return base->mask(shadingData);
	}
};