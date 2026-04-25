

#include "GEMLoader.h"
#include "Renderer.h"
#include "SceneLoader.h"
#define NOMINMAX
#include "GamesEngineeringBase.h"
#include <unordered_map>
#include <random>
#include <cmath>
#include <iomanip>


bool nearlyEqual(float a, float b, float eps = 1e-4f)
{
	return std::fabs(a - b) < eps;
}

void testSamplingDistributions()
{
	std::cout << "Running tests for SamplingDistributions" << std::endl;

	const int N = 100000;
	std::mt19937 rng(12345);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// -----------------------------
	// Test 3.1: uniform hemisphere
	// -----------------------------
	{
		bool ok = true;
		float avgZ = 0.0f;

		for (int i = 0; i < N; i++)
		{
			float r1 = dist(rng);
			float r2 = dist(rng);
			Vec3 v = SamplingDistributions::uniformSampleHemisphere(r1, r2);

			float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
			if (v.z < 0.0f || !nearlyEqual(len, 1.0f, 1e-3f))
			{
				ok = false;
				std::cout << "[Uniform Hemisphere] Invalid sample: ("
					<< v.x << ", " << v.y << ", " << v.z << "), len=" << len << std::endl;
				break;
			}
			avgZ += v.z;
		}

		avgZ /= N;

		std::cout << "[Test 3.1] Uniform Hemisphere validity" << std::endl;
		std::cout << "Expected: all samples on unit hemisphere, avgZ ~= 0.5" << std::endl;
		std::cout << "Actual: ok=" << ok << ", avgZ=" << avgZ << std::endl;
		std::cout << std::endl;
	}

	// -----------------------------
	// Test 3.2: cosine hemisphere
	// -----------------------------
	{
		bool ok = true;
		float avgZ = 0.0f;

		for (int i = 0; i < N; i++)
		{
			float r1 = dist(rng);
			float r2 = dist(rng);
			Vec3 v = SamplingDistributions::cosineSampleHemisphere(r1, r2);

			float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
			if (v.z < 0.0f || !nearlyEqual(len, 1.0f, 1e-3f))
			{
				ok = false;
				std::cout << "[Cosine Hemisphere] Invalid sample: ("
					<< v.x << ", " << v.y << ", " << v.z << "), len=" << len << std::endl;
				break;
			}
			avgZ += v.z;
		}

		avgZ /= N;

		std::cout << "[Test 3.2] Cosine Hemisphere validity" << std::endl;
		std::cout << "Expected: all samples on unit hemisphere, avgZ ~= 2/3 = 0.6667" << std::endl;
		std::cout << "Actual: ok=" << ok << ", avgZ=" << avgZ << std::endl;
		std::cout << std::endl;
	}

	// -----------------------------
	// Test 3.3: uniform sphere
	// -----------------------------
	{
		bool ok = true;
		float avgZ = 0.0f;

		for (int i = 0; i < N; i++)
		{
			float r1 = dist(rng);
			float r2 = dist(rng);
			Vec3 v = SamplingDistributions::uniformSampleSphere(r1, r2);

			float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
			if (!nearlyEqual(len, 1.0f, 1e-3f))
			{
				ok = false;
				std::cout << "[Uniform Sphere] Invalid sample: ("
					<< v.x << ", " << v.y << ", " << v.z << "), len=" << len << std::endl;
				break;
			}
			avgZ += v.z;
		}

		avgZ /= N;

		std::cout << "[Test 3.3] Uniform Sphere validity" << std::endl;
		std::cout << "Expected: all samples on unit sphere, avgZ ~= 0" << std::endl;
		std::cout << "Actual: ok=" << ok << ", avgZ=" << avgZ << std::endl;
		std::cout << std::endl;
	}
}

void runTests()
{
	// Add test code here
	std::cout << "Running tests for plane" << std::endl;
	Plane p;
	Vec3 normal(0.f, 1.f, 0.f);
	p.init(normal, 0.f);

	// Test 1.1: should hit
	{
		Ray r(Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
		float t = -1.0f;
		bool hit = p.rayIntersect(r, t);

		std::cout << "[Test 1.1] Expected: hit=true, t=1" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << std::endl;

		if (hit)
		{
			Vec3 pt = r.at(t);
			std::cout << "Intersection point: ("
				<< pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
		}
		std::cout << std::endl;
	}

	// Test 1.2: should miss
	{
		Ray r(Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
		float t = -1.0f;
		bool hit = p.rayIntersect(r, t);
		std::cout << "[Test 1.2] Expected: hit=false" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << std::endl;
		std::cout << std::endl;
	}

	// Test 1.3: should miss (parallel)
	{
		Ray r(Vec3(0.0f, 1.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
		float t = -1.0f;
		bool hit = p.rayIntersect(r, t);
		std::cout << "[Test 1.3] Expected: hit=false" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << std::endl;
		std::cout << std::endl;
	}

	// Test 1.4: should hit
	{
		Ray r(Vec3(0.2f, 1.0f, 0.2f), Vec3(-1.0f, -1.0f, -1.0f));
		p.init(normal, 0.0f);
		float t = -1.0f;
		bool hit = p.rayIntersect(r, t);

		std::cout << "[Test 1.4] Expected: hit=true, t=1" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << std::endl;

		if (hit)
		{
			Vec3 pt = r.at(t);
			std::cout << "Intersection point: ("
				<< pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
		}
		std::cout << std::endl;
	}

	// Test 1.5: should miss
	{
		Ray r(Vec3(-0.2f, -1.0f, -0.2f), Vec3(-1.0f, -1.0f, -1.0f));
		p.init(normal, 0.0f);
		float t = -1.0f;
		bool hit = p.rayIntersect(r, t);

		std::cout << "[Test 1.5] Expected: hit=false" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << std::endl;

		std::cout << std::endl;
	}

	// Add test code here
	std::cout << "Running tests for triangle" << std::endl;
	Vertex v0, v1, v2;

	v0.p = Vec3(0.0f, 0.0f, 0.0f);
	v1.p = Vec3(1.0f, 0.0f, 0.0f);
	v2.p = Vec3(0.0f, 0.0f, 1.0f);

	v0.normal = Vec3(0.0f, 1.0f, 0.0f);
	v1.normal = Vec3(0.0f, 1.0f, 0.0f);
	v2.normal = Vec3(0.0f, 1.0f, 0.0f);

	v0.u = 0.0f; v0.v = 0.0f;
	v1.u = 1.0f; v1.v = 0.0f;
	v2.u = 0.0f; v2.v = 1.0f;

	Triangle tri;
	tri.init(v0, v1, v2, 0);

	// Test 2.1 should hit
	{
		Ray r(Vec3(0.0f, 1.0f, 0.0f), Vec3(-0.5f, -1.0f, -0.5f));
		float t = -1.0f, u = -1.0f, v = -1.0f;
		bool hit = tri.rayIntersect(r, t, u, v);

		std::cout << "[Test 2.1] Expected: hit=true" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << ", u=" << u << ", v=" << v << std::endl;

		if (hit)
		{
			Vec3 pt = r.at(t);
			float w = 1.0f - u - v;
			std::cout << "Intersection point: ("
				<< pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
			std::cout << "Barycentric: w=" << w << ", u=" << u << ", v=" << v << std::endl;
		}
		std::cout << std::endl;
	}

	// Test 2.2: hit plane but outside triangle
	{
		Ray r(Vec3(1.2f, 1.0f, 1.2f), Vec3(0.0f, -1.0f, 0.0f));
		float t = -1.0f, u = -1.0f, v = -1.0f;
		bool hit = tri.rayIntersect(r, t, u, v);

		std::cout << "[Test 2.2] Expected: hit=false" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << ", u=" << u << ", v=" << v << std::endl;
		std::cout << std::endl;
	}

	// Test 2.3: parallel to triangle plane
	{
		Ray r(Vec3(0.25f, 1.0f, 0.25f), Vec3(1.0f, 0.0f, 0.0f));
		float t = -1.0f, u = -1.0f, v = -1.0f;
		bool hit = tri.rayIntersect(r, t, u, v);

		std::cout << "[Test 2.3] Expected: hit=false" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << ", u=" << u << ", v=" << v << std::endl;
		std::cout << std::endl;
	}

	// Test 2.4: triangle is behind ray origin
	{
		Ray r(Vec3(0.25f, -1.0f, 0.25f), Vec3(-1.0f, -1.0f, -1.0f));
		float t = -1.0f, u = -1.0f, v = -1.0f;
		bool hit = tri.rayIntersect(r, t, u, v);

		std::cout << "[Test 2.4] Expected: hit=false" << std::endl;
		std::cout << "Actual: hit=" << hit << ", t=" << t << ", u=" << u << ", v=" << v << std::endl;
		std::cout << std::endl;
	}


}

int main(int argc, char *argv[])
{
	//runTests();
	//testSamplingDistributions();
	//return 0;
	
	// Initialize default parameters
	std::string sceneName = "dining-room";
	std::string filename = "GI.hdr";
	unsigned int SPP = 512;

	if (argc > 1)
	{
		std::unordered_map<std::string, std::string> args;
		for (int i = 1; i < argc; ++i)
		{
			std::string arg = argv[i];
			if (!arg.empty() && arg[0] == '-')
			{
				std::string argName = arg;
				if (i + 1 < argc)
				{
					std::string argValue = argv[++i];
					args[argName] = argValue;
				} else
				{
					std::cerr << "Error: Missing value for argument '" << arg << "'\n";
				}
			} else
			{
				std::cerr << "Warning: Ignoring unexpected argument '" << arg << "'\n";
			}
		}
		for (const auto& pair : args)
		{
			if (pair.first == "-scene")
			{
				sceneName = pair.second;
			}
			if (pair.first == "-outputFilename")
			{
				filename = pair.second;
			}
			if (pair.first == "-SPP")
			{
				SPP = stoi(pair.second);
			}
		}
	}
	Scene* scene = loadScene(sceneName);
	GamesEngineeringBase::Window canvas;
	canvas.create((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, "Tracer", false);
	RayTracer rt;
	rt.init(scene, &canvas);
	bool running = true;
	GamesEngineeringBase::Timer timer;
	while (running)
	{
		canvas.checkInput();
		canvas.clear();
		if (canvas.keyPressed(VK_ESCAPE))
		{
			break;
		}
		if (canvas.keyPressed('W'))
		{
			viewcamera.forward();
			rt.clear();
		}
		if (canvas.keyPressed('S'))
		{
			viewcamera.back();
			rt.clear();
		}
		if (canvas.keyPressed('A'))
		{
			viewcamera.left();
			rt.clear();
		}
		if (canvas.keyPressed('D'))
		{
			viewcamera.right();
			rt.clear();
		}
		if (canvas.keyPressed('E'))
		{
			viewcamera.flyUp();
			rt.clear();
		}
		if (canvas.keyPressed('Q'))
		{
			viewcamera.flyDown();
			rt.clear();
		}
		// Time how long a render call takes
		timer.reset();
		rt.render();
		float t = timer.dt();
		// Write
		std::cout << t << std::endl;
		if (canvas.keyPressed('P'))
		{
			rt.saveHDR(filename);
		}
		if (canvas.keyPressed('L'))
		{
			size_t pos = filename.find_last_of('.');
			std::string ldrFilename = filename.substr(0, pos) + ".png";
			rt.savePNG(ldrFilename);
		}
		if (SPP == rt.getSPP())
		{
			rt.saveHDR(filename);
			break;
		}
		canvas.present();
	}
	return 0;
}