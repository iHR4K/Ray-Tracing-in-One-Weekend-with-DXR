//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "stdafx.h"
#include "D3D12RaytracingInOneWeekend.h"
#include "DirectXRaytracingHelper.h"
#include "CompiledShaders\Raytracing.hlsl.h"

#include <random>

using namespace std;
using namespace DX;

const wchar_t* D3D12RaytracingInOneWeekend::c_hitGroupName = L"MyHitGroup";
const wchar_t* D3D12RaytracingInOneWeekend::c_raygenShaderName = L"MyRaygenShader";
const wchar_t* D3D12RaytracingInOneWeekend::c_closestHitShaderName = L"MyClosestHitShader";
const wchar_t* D3D12RaytracingInOneWeekend::c_missShaderName = L"MyMissShader";

namespace
{
	// FROM BOOK: 3D Game Programming With DX12 - amazing one, worth to check
	std::pair<std::vector<Vertex>, std::vector<int>>
		CreateSphere(float radius, uint32_t sliceCount, uint32_t stackCount)
	{
		Vertex topVertex({ 0.0f, +radius, 0.0f }, 0.0, { 0.0f, 1.0f, 0.0f }, 0.0);

		std::vector<Vertex> vertices;
		vertices.push_back(topVertex);

		float phiStep = XM_PI / stackCount;
		float thetaStep = 2.0f * XM_PI / sliceCount;

		// Compute vertices for each stack ring (do not count the poles as rings).
		for (uint32_t i = 1; i <= stackCount - 1; ++i)
		{
			float phi = i * phiStep;

			// Vertices of ring.
			for (uint32_t j = 0; j <= sliceCount; ++j)
			{
				float theta = j * thetaStep;

				Vertex v{ };

				// spherical to cartesian
				v.position.x = radius * sinf(phi) * cosf(theta);
				v.position.y = radius * cosf(phi);
				v.position.z = radius * sinf(phi) * sinf(theta);

				XMVECTOR p = XMLoadFloat3(&v.position);
				XMStoreFloat3(&v.normal, XMVector3Normalize(p));

				vertices.push_back(v);
			}
		}

		Vertex bottomVertex({ 0.0f, -radius, 0.0f }, 0.0, { 0.0f, -1.0f, 0.0f }, 0.0);
		vertices.push_back(bottomVertex);



		std::vector<int> indices;

		//
		// Compute indices for top stack.  The top stack was written first to the vertex buffer
		// and connects the top pole to the first ring.
		//
		for (uint32_t i = 1; i <= sliceCount; ++i)
		{
			indices.push_back(0);
			indices.push_back(i + 1);
			indices.push_back(i);
		}

		//
		// Compute indices for inner stacks (not connected to poles).
		//
		uint32_t baseIndex = 1;
		uint32_t ringVertexCount = sliceCount + 1;
		for (uint32_t i = 0; i < stackCount - 2; ++i)
		{
			for (uint32_t j = 0; j < sliceCount; ++j)
			{
				indices.push_back(baseIndex + i * ringVertexCount + j);
				indices.push_back(baseIndex + i * ringVertexCount + j + 1);
				indices.push_back(baseIndex + (i + 1) * ringVertexCount + j);

				indices.push_back(baseIndex + (i + 1) * ringVertexCount + j);
				indices.push_back(baseIndex + i * ringVertexCount + j + 1);
				indices.push_back(baseIndex + (i + 1) * ringVertexCount + j + 1);
			}
		}

		//
		// Compute indices for bottom stack.  The bottom stack was written last to the vertex buffer
		// and connects the bottom pole to the bottom ring.
		//

		// South pole vertex was added last.
		uint32_t southPoleIndex = (uint32_t)vertices.size() - 1;

		// Offset the indices to the index of the first vertex in the last ring.
		baseIndex = southPoleIndex - ringVertexCount;

		for (uint32_t i = 0; i < sliceCount; ++i)
		{
			indices.push_back(southPoleIndex);
			indices.push_back(baseIndex + i);
			indices.push_back(baseIndex + i + 1);
		}

		return { vertices, indices };
	}
}

D3D12RaytracingInOneWeekend::D3D12RaytracingInOneWeekend(UINT width, UINT height, std::wstring name) :
	DXSample(width, height, name),
	m_raytracingOutputResourceUAVDescriptorHeapIndex(UINT_MAX),
	m_curRotationAngleRad(0.0f)
{
	UpdateForSizeChange(width, height);
}

void D3D12RaytracingInOneWeekend::OnInit()
{
	m_deviceResources = std::make_unique<DeviceResources>(
		DXGI_FORMAT_R8G8B8A8_UNORM,
		DXGI_FORMAT_UNKNOWN,
		FrameCount,
		D3D_FEATURE_LEVEL_11_0,
		// Sample shows handling of use cases with tearing support, which is OS dependent and has been supported since TH2.
		// Since the sample requires build 1809 (RS5) or higher, we don't need to handle non-tearing cases.
		DeviceResources::c_RequireTearingSupport,
		m_adapterIDoverride
	);
	m_deviceResources->RegisterDeviceNotify(this);
	m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
	m_deviceResources->InitializeDXGIAdapter();

	ThrowIfFalse(IsDirectXRaytracingSupported(m_deviceResources->GetAdapter()),
		L"ERROR: DirectX Raytracing is not supported by your OS, GPU and/or driver.\n\n");

	m_deviceResources->CreateDeviceResources();
	m_deviceResources->CreateWindowSizeDependentResources();

	InitializeScene();

	CreateDeviceDependentResources();
	CreateWindowSizeDependentResources();
}

// Update camera matrices passed into the shader.
void D3D12RaytracingInOneWeekend::UpdateCameraMatrices()
{
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	m_frameCB[frameIndex].cameraPosition = m_eye;
	float fovAngleY = 20.0f;
	XMMATRIX view = XMMatrixLookAtRH(m_eye, m_at, m_up);
	XMMATRIX proj = XMMatrixPerspectiveFovRH(XMConvertToRadians(fovAngleY), m_aspectRatio, 0.1f, 10000.0f);
	XMMATRIX viewProj = view * proj;

	m_frameCB[frameIndex].projectionToWorld = XMMatrixInverse(nullptr, proj);
	m_frameCB[frameIndex].modelViewInverse = XMMatrixInverse(nullptr, view);
}

// Initialize scene rendering parameters.
void D3D12RaytracingInOneWeekend::InitializeScene()
{
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	// Setup camera.
	{
		// Initialize the view and projection inverse matrices.
		m_eye = { 13.0f, 2.0f, 3.0f, 1.0f };
		m_at = { 0.0f, 0.0f, 0.0f, 1.0f };
		m_up = { 0.0f, 1.0f, 0.0f, 1.0f };

		UpdateCameraMatrices();
	}

	// Apply the initial values to all frames' buffer instances.
	for (auto& sceneCB : m_frameCB)
	{
		sceneCB = m_frameCB[frameIndex];
	}
}

// Create constant buffers.
void D3D12RaytracingInOneWeekend::CreateConstantBuffers()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameCount = m_deviceResources->GetBackBufferCount();

	// Create the constant buffer memory and map the CPU and GPU addresses
	const D3D12_HEAP_PROPERTIES uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

	// Allocate one constant buffer per frame, since it gets updated every frame.
	size_t cbSize = frameCount * sizeof(AlignedSceneConstantBuffer);
	const D3D12_RESOURCE_DESC constantBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(cbSize);

	ThrowIfFailed(device->CreateCommittedResource(
		&uploadHeapProperties,
		D3D12_HEAP_FLAG_NONE,
		&constantBufferDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_perFrameConstants)));

	// Map the constant buffer and cache its heap pointers.
	// We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay.
	CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
	ThrowIfFailed(m_perFrameConstants->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedConstantData)));
}

// Create resources that depend on the device.
void D3D12RaytracingInOneWeekend::CreateDeviceDependentResources()
{
	// Initialize raytracing pipeline.

	// Create raytracing interfaces: raytracing device and commandlist.
	CreateRaytracingInterfaces();

	// Create root signatures for the shaders.
	CreateRootSignatures();

	// Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
	CreateRaytracingPipelineStateObject();

	// Create a heap for descriptors.
	CreateDescriptorHeap();

	// Build geometry to be used in the sample.
	BuildGeometry();

	// Build raytracing acceleration structures from the generated geometry.
	BuildAccelerationStructures();


	// Create constant buffers for the geometry and the scene.
	CreateConstantBuffers();

	// Build shader tables, which define shaders and their local root arguments.
	BuildShaderTables();

	// Create an output 2D texture to store the raytracing result to.
	CreateRaytracingOutputResource();
}

void D3D12RaytracingInOneWeekend::SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig)
{
	auto device = m_deviceResources->GetD3DDevice();
	ComPtr<ID3DBlob> blob;
	ComPtr<ID3DBlob> error;

	ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), error ? static_cast<wchar_t*>(error->GetBufferPointer()) : nullptr);
	ThrowIfFailed(device->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&(*rootSig))));
}

void D3D12RaytracingInOneWeekend::CreateRootSignatures()
{
	auto device = m_deviceResources->GetD3DDevice();

	// Global Root Signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	{
		CD3DX12_DESCRIPTOR_RANGE ranges[2]; // Perfomance TIP: Order from most frequent to least frequent.
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1);  // 2 static index and vertex buffers.

		CD3DX12_ROOT_PARAMETER rootParameters[GlobalRootSignatureParams::Count];
		rootParameters[GlobalRootSignatureParams::OutputViewSlot].InitAsDescriptorTable(1, &ranges[0]);
		rootParameters[GlobalRootSignatureParams::AccelerationStructureSlot].InitAsShaderResourceView(0);
		rootParameters[GlobalRootSignatureParams::SceneConstantSlot].InitAsConstantBufferView(0);
		rootParameters[GlobalRootSignatureParams::VertexBuffersSlot].InitAsDescriptorTable(1, &ranges[1]);
		CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
		SerializeAndCreateRaytracingRootSignature(globalRootSignatureDesc, &m_raytracingGlobalRootSignature);
	}

	// Local Root Signature
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	{
		CD3DX12_ROOT_PARAMETER rootParameters[LocalRootSignatureParams::Count];
		// TODO: remove this 
		rootParameters[LocalRootSignatureParams::MeshBufferSlot].InitAsConstants(sizeof(MeshBuffer), 1);
		CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
		localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
		SerializeAndCreateRaytracingRootSignature(localRootSignatureDesc, &m_raytracingLocalRootSignature);
	}
}

// Create raytracing device and command list.
void D3D12RaytracingInOneWeekend::CreateRaytracingInterfaces()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();

	ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_dxrDevice)), L"Couldn't get DirectX Raytracing interface for the device.\n");
	ThrowIfFailed(commandList->QueryInterface(IID_PPV_ARGS(&m_dxrCommandList)), L"Couldn't get DirectX Raytracing interface for the command list.\n");
}

// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void D3D12RaytracingInOneWeekend::CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
	// Ray gen and miss shaders in this sample are not using a local root signature and thus one is not associated with them.

	// Local root signature to be used in a hit group.
	auto localRootSignature = raytracingPipeline->CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
	localRootSignature->SetRootSignature(m_raytracingLocalRootSignature.Get());
	// Define explicit shader association for the local root signature. 
	{
		auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
		rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
		rootSignatureAssociation->AddExport(c_hitGroupName);
	}
}

// Create a raytracing pipeline state object (RTPSO).
// An RTPSO represents a full set of shaders reachable by a DispatchRays() call,
// with all configuration options resolved, such as local signatures and other state.
void D3D12RaytracingInOneWeekend::CreateRaytracingPipelineStateObject()
{
	// Create 7 subobjects that combine into a RTPSO:
	// Subobjects need to be associated with DXIL exports (i.e. shaders) either by way of default or explicit associations.
	// Default association applies to every exported shader entrypoint that doesn't have any of the same type of subobject associated with it.
	// This simple sample utilizes default shader association except for local root signature subobject
	// which has an explicit association specified purely for demonstration purposes.
	// 1 - DXIL library
	// 1 - Triangle hit group
	// 1 - Shader config
	// 2 - Local root signature and association
	// 1 - Global root signature
	// 1 - Pipeline config
	CD3DX12_STATE_OBJECT_DESC raytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };


	// DXIL library
	// This contains the shaders and their entrypoints for the state object.
	// Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
	auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
	D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void*)g_pRaytracing, ARRAYSIZE(g_pRaytracing));
	lib->SetDXILLibrary(&libdxil);
	// Define which shader exports to surface from the library.
	// If no shader exports are defined for a DXIL library subobject, all shaders will be surfaced.
	// In this sample, this could be ommited for convenience since the sample uses all shaders in the library. 
	{
		lib->DefineExport(c_raygenShaderName);
		lib->DefineExport(c_closestHitShaderName);
		lib->DefineExport(c_missShaderName);
	}

	// Triangle hit group
	// A hit group specifies closest hit, any hit and intersection shaders to be executed when a ray intersects the geometry's triangle/AABB.
	// In this sample, we only use triangle geometry with a closest hit shader, so others are not set.
	auto hitGroup = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
	hitGroup->SetClosestHitShaderImport(c_closestHitShaderName);
	hitGroup->SetHitGroupExport(c_hitGroupName);
	hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

	// Shader config
	// Defines the maximum sizes in bytes for the ray payload and attribute structure.
	auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
	UINT payloadSize = sizeof(XMFLOAT4) + sizeof(UINT) + sizeof(XMFLOAT4);// float4 pixelColor
	UINT attributeSize = sizeof(XMFLOAT2);  // float2 barycentrics
	shaderConfig->Config(payloadSize, attributeSize);

	// Local root signature and shader association
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	CreateLocalRootSignatureSubobjects(&raytracingPipeline);

	// Global root signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
	globalRootSignature->SetRootSignature(m_raytracingGlobalRootSignature.Get());

	// Pipeline config
	// Defines the maximum TraceRay() recursion depth.
	auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
	// PERFOMANCE TIP: Set max recursion depth as low as needed 
	// as drivers may apply optimization strategies for low recursion depths.
	UINT maxRecursionDepth = 1; // ~ primary rays only. 
	pipelineConfig->Config(maxRecursionDepth);

#if _DEBUG
	PrintStateObjectDesc(raytracingPipeline);
#endif

	// Create the state object.
	ThrowIfFailed(m_dxrDevice->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_dxrStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
}

// Create 2D output texture for raytracing.
void D3D12RaytracingInOneWeekend::CreateRaytracingOutputResource()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

	// Create the output resource. The dimensions and format should match the swap-chain.
	auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(backbufferFormat, m_width, m_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	ThrowIfFailed(device->CreateCommittedResource(
		&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &uavDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_raytracingOutput)));
	NAME_D3D12_OBJECT(m_raytracingOutput);

	D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
	m_raytracingOutputResourceUAVDescriptorHeapIndex = AllocateDescriptor(&uavDescriptorHandle, m_raytracingOutputResourceUAVDescriptorHeapIndex);
	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc{ };
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device->CreateUnorderedAccessView(m_raytracingOutput.Get(), nullptr, &UAVDesc, uavDescriptorHandle);
	m_raytracingOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), m_raytracingOutputResourceUAVDescriptorHeapIndex, m_descriptorSize);
}

void D3D12RaytracingInOneWeekend::CreateDescriptorHeap()
{
	auto device = m_deviceResources->GetD3DDevice();

	D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{ };
	// Allocate a heap for 3 descriptors:
	// 2 - vertex and index buffer SRVs
	// 1 - raytracing output texture SRV
	descriptorHeapDesc.NumDescriptors = 3;
	descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	descriptorHeapDesc.NodeMask = 0;
	device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&m_descriptorHeap));
	NAME_D3D12_OBJECT(m_descriptorHeap);

	m_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

inline double random_double(double from, double to)
{
	static std::uniform_real_distribution<double> distribution(from, to);
	static std::mt19937 generator;
	return distribution(generator);
}

inline double random_double()
{
	return random_double(0, 1.0);
}

inline float random_float(float from, float to)
{
	static std::uniform_real_distribution<float> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline float random_float()
{
	return random_float(0.0, 1.0);
}

namespace
{
	XMVECTOR randomColor()
	{
		return XMVECTOR{ random_float(), random_float(), random_float() };
	}
}

// Build geometry used in the sample.
void D3D12RaytracingInOneWeekend::BuildGeometry()
{
	auto device = m_deviceResources->GetD3DDevice();

	int const cLambertian{ 0 };
	int const cMetallic{ 1 };
	int const cDielectric{ 2 };

	int const sliceCount{ 16 };
	int const stackCount{ 32 };
	{
		auto sphereGeometry = CreateSphere(1000, sliceCount, 512);

		m_geometry.push_back(Geometry{ .vertices = sphereGeometry.first, .indices = sphereGeometry.second,
			.albedo = XMFLOAT4{ 0.5, 0.5, 0.5, 1.0 },
			.materialId = cLambertian,
			.transform = XMMatrixTranspose(XMMatrixTranslation(0, -1000, 0)) });
	}

	{
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				auto choose_mat = random_double();
				XMVECTOR center{ a + 0.9 * random_double(), 0.2, b + 0.9 * random_double() };
				auto length = XMVector3Length(XMVectorSubtract(center, XMVECTOR{ 4, 0.2, 0 }));
				if (XMVectorGetX(length) > 0.9f)
				{
					XMVECTOR materialParameter;
					int materialType{ 0 };

					if (choose_mat < 0.8)
					{
						// diffuse
						materialType = cLambertian;
						auto albedo = randomColor() * randomColor();
						materialParameter = albedo;
					}
					else if (choose_mat < 0.95)
					{
						// metal
						materialType = cMetallic;
						auto fuzz = random_float(0, 0.5);
						materialParameter = XMVECTOR{ random_float(0.5, 1), random_float(0.5, 1), random_float(0.5, 1), fuzz };
					}
					else
					{
						// glass
						materialType = cDielectric;
						materialParameter = XMVECTOR{ 1.5, 1.5, 1.5, 1.5 };
					}

					auto sphereGeometry = CreateSphere(0.2, sliceCount, stackCount);
					XMFLOAT4 float4Material;
					XMStoreFloat4(&float4Material, materialParameter);

					m_geometry.push_back(Geometry{ .vertices = sphereGeometry.first, .indices = sphereGeometry.second,
						.albedo = float4Material,
						.materialId = materialType,
						.transform = XMMatrixTranspose(XMMatrixTranslationFromVector(center)) });
				}
			}
		}
		{
			auto sphereGeometry = CreateSphere(1.0, sliceCount, stackCount);

			m_geometry.push_back(Geometry{ .vertices = sphereGeometry.first, .indices = sphereGeometry.second,
				.albedo = { 1.5, 1.5, 1.5, 1.5 },
				.materialId = cDielectric,
				.transform = XMMatrixTranspose(XMMatrixTranslation(0, 1, 0)) });
		}

		{
			auto sphereGeometry = CreateSphere(1.0, sliceCount, stackCount);

			m_geometry.push_back(Geometry{ .vertices = sphereGeometry.first, .indices = sphereGeometry.second,
				.albedo = { 0.4, 0.2, 0.1, 0.0 },
				.materialId = cLambertian,
				.transform = XMMatrixTranspose(XMMatrixTranslation(-4, 1, 0)) });
		}

		{
			auto sphereGeometry = CreateSphere(1.0, sliceCount, stackCount);

			m_geometry.push_back(Geometry{ .vertices = sphereGeometry.first, .indices = sphereGeometry.second,
				.albedo = { 0.7, 0.6, 0.5, 0.0 },
				.materialId = cMetallic,
				.transform = XMMatrixTranspose(XMMatrixTranslation(4, 1, 0)) });
		}
	}


	size_t totalNumIndices = 0;
	size_t totalNumVertices = 0;

	for (auto const& geometry : m_geometry)
	{
		totalNumIndices += geometry.indices.size();
		totalNumVertices += geometry.vertices.size();
	}

	std::vector<UINT> indices(totalNumIndices);
	std::vector<Vertex> vertices(totalNumVertices);
	size_t vertexOffset = 0;
	size_t indexOffset = 0;
	for (auto& geom : m_geometry)
	{
		memcpy(&vertices[vertexOffset], geom.vertices.data(), geom.vertices.size() * sizeof(geom.vertices[0]));
		memcpy(indices.data() + indexOffset, geom.indices.data(), geom.indices.size() * sizeof(geom.indices[0]));
		geom.indicesOffsetInBytes = indexOffset * sizeof(geom.indices[0]);
		geom.verticesOffsetInBytes = vertexOffset * sizeof(geom.vertices[0]);

		vertexOffset += geom.vertices.size();
		indexOffset += geom.indices.size();
	}
	AllocateUploadBuffer(device, vertices.data(), vertices.size() * sizeof(Vertex), &m_vertexBuffer.resource);
	AllocateUploadBuffer(device, indices.data(), indices.size() * sizeof(UINT), &m_indexBuffer.resource);


	// Vertex buffer is passed to the shader along with index buffer as a descriptor table.
	// Vertex buffer descriptor must follow index buffer descriptor in the descriptor heap.
	UINT descriptorIndexIB = CreateBufferSRV(&m_indexBuffer, indices.size(), sizeof(INT));
	UINT descriptorIndexVB = CreateBufferSRV(&m_vertexBuffer, vertices.size(), sizeof(vertices[0]));
	ThrowIfFalse(descriptorIndexVB == descriptorIndexIB + 1, L"Vertex Buffer descriptor index must follow that of Index Buffer descriptor index!");
}

// Build acceleration structures needed for raytracing.
void D3D12RaytracingInOneWeekend::BuildAccelerationStructures()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto commandQueue = m_deviceResources->GetCommandQueue();
	auto commandAllocator = m_deviceResources->GetCommandAllocator();

	std::vector<ComPtr<ID3D12Resource>> scratches; // just to expand lifetime
	std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDescriptors;
	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;

	// Reset the command list for the acceleration structure construction.
	commandList->Reset(commandAllocator, nullptr);

	int i{ 0 };
	for (auto& geometry : m_geometry)
	{
		auto& m_bottomLevelAccelerationStructure = geometry.m_bottomLevelAccelerationStructure;

		D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc{ };
		geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
		geometryDesc.Triangles.VertexBuffer.StartAddress = m_vertexBuffer.resource->GetGPUVirtualAddress() + geometry.verticesOffsetInBytes;
		geometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(geometry.vertices[0]);
		geometryDesc.Triangles.VertexCount = static_cast<UINT>(geometry.vertices.size());
		geometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;

		geometryDesc.Triangles.IndexBuffer = m_indexBuffer.resource->GetGPUVirtualAddress() + geometry.indicesOffsetInBytes;
		geometryDesc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
		geometryDesc.Triangles.IndexCount = static_cast<UINT>(geometry.indices.size());

		geometryDescs.push_back(geometryDesc);

		// Get required sizes for an acceleration structure.
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc{ };
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& bottomLevelInputs = bottomLevelBuildDesc.Inputs;
		bottomLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		bottomLevelInputs.Flags = buildFlags;
		bottomLevelInputs.NumDescs = 1;
		bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
		bottomLevelInputs.pGeometryDescs = &geometryDesc;

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo{ };
		m_dxrDevice->GetRaytracingAccelerationStructurePrebuildInfo(&bottomLevelInputs, &bottomLevelPrebuildInfo);
		ThrowIfFalse(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);

		D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
		AllocateUAVBuffer(device, bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, &m_bottomLevelAccelerationStructure, initialResourceState, L"BottomLevelAccelerationStructure");

		D3D12_RAYTRACING_INSTANCE_DESC instanceDesc{ };
		memcpy(instanceDesc.Transform, geometry.transform.r, 12 * sizeof(float));
		instanceDesc.InstanceMask = 1;
		instanceDesc.InstanceID = i;
		instanceDesc.InstanceContributionToHitGroupIndex = i;
		instanceDesc.AccelerationStructure = m_bottomLevelAccelerationStructure->GetGPUVirtualAddress();

		instanceDescriptors.push_back(instanceDesc);

		ComPtr<ID3D12Resource> scratchResource;
		AllocateUAVBuffer(device, bottomLevelPrebuildInfo.ScratchDataSizeInBytes, &scratchResource, D3D12_RESOURCE_STATE_COMMON, L"ScratchResource");
		scratches.push_back(scratchResource); // extend up to execute

		// Bottom Level Acceleration Structure desc
		{
			bottomLevelBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
			bottomLevelBuildDesc.DestAccelerationStructureData = m_bottomLevelAccelerationStructure->GetGPUVirtualAddress();
		}

		// Build acceleration structure.
		m_dxrCommandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);
		auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_bottomLevelAccelerationStructure.Get());
		commandList->ResourceBarrier(1, &barrier);

		++i;
	}

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc{ };
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& topLevelInputs = topLevelBuildDesc.Inputs;
	topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	topLevelInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
	topLevelInputs.NumDescs = static_cast<UINT>(geometryDescs.size());
	topLevelInputs.pGeometryDescs = geometryDescs.data();
	topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo{ };
	m_dxrDevice->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);
	ThrowIfFalse(topLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);


	ComPtr<ID3D12Resource> m_scratchResource;

	D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
	AllocateUAVBuffer(device, topLevelPrebuildInfo.ScratchDataSizeInBytes, &m_scratchResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource");
	AllocateUAVBuffer(device, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, &m_topLevelAccelerationStructure, initialResourceState, L"TopLevelAccelerationStructure");

	// Top Level Acceleration Structure desc
	ComPtr<ID3D12Resource> m_instanceDescs;
	AllocateUploadBuffer(m_deviceResources->GetD3DDevice(), instanceDescriptors.data(), sizeof(instanceDescriptors[0]) * instanceDescriptors.size(),
		&m_instanceDescs, L"InstanceDescs");

	topLevelBuildDesc.DestAccelerationStructureData = m_topLevelAccelerationStructure->GetGPUVirtualAddress();
	topLevelBuildDesc.ScratchAccelerationStructureData = m_scratchResource->GetGPUVirtualAddress();
	topLevelBuildDesc.Inputs.InstanceDescs = m_instanceDescs->GetGPUVirtualAddress();

	m_dxrCommandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);

	// Kick off acceleration structure construction.
	m_deviceResources->ExecuteCommandList();

	// Wait for GPU to finish as the locally created temporary GPU resources will get released once we go out of scope.
	m_deviceResources->WaitForGpu();
}

// Build shader tables.
// This encapsulates all shader records - shaders and the arguments for their local root signatures.
void D3D12RaytracingInOneWeekend::BuildShaderTables()
{
	auto device = m_deviceResources->GetD3DDevice();

	void* rayGenShaderIdentifier;
	void* missShaderIdentifier;
	void* hitGroupShaderIdentifier;

	auto GetShaderIdentifiers = [&](auto* stateObjectProperties)
		{
			rayGenShaderIdentifier = stateObjectProperties->GetShaderIdentifier(c_raygenShaderName);
			missShaderIdentifier = stateObjectProperties->GetShaderIdentifier(c_missShaderName);
			hitGroupShaderIdentifier = stateObjectProperties->GetShaderIdentifier(c_hitGroupName);
		};

	// Get shader identifiers.
	UINT shaderIdentifierSize;
	{
		ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
		ThrowIfFailed(m_dxrStateObject.As(&stateObjectProperties));
		GetShaderIdentifiers(stateObjectProperties.Get());
		shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	}

	// Ray gen shader table
	{
		UINT numShaderRecords = 1;
		UINT shaderRecordSize = shaderIdentifierSize;
		ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable");
		rayGenShaderTable.push_back(ShaderRecord(rayGenShaderIdentifier, shaderIdentifierSize));
		m_rayGenShaderTable = rayGenShaderTable.GetResource();
	}

	// Miss shader table
	{
		UINT numShaderRecords = 1;
		UINT shaderRecordSize = shaderIdentifierSize;
		ShaderTable missShaderTable(device, numShaderRecords, shaderRecordSize, L"MissShaderTable");
		missShaderTable.push_back(ShaderRecord(missShaderIdentifier, shaderIdentifierSize));
		m_missShaderTable = missShaderTable.GetResource();
	}

	// This is very important moment which we usually can't see in "intro-demos", 
	// I especcialy left this in very simple form so you can track how to work with a few shader table records,
	// we need this toi populate local root arguments in shader
	// Hit group shader table
	{
		UINT verticesOffset = 0;
		UINT indicesOffset = 0;

		std::vector<MeshBuffer> args;

		for (UINT i = 0; i < m_geometry.size(); ++i)
		{
			MeshBuffer meshBuffer{ };
			meshBuffer.meshId = i;
			meshBuffer.materialId = m_geometry[i].materialId;
			meshBuffer.albedo = m_geometry[i].albedo;
			meshBuffer.verticesOffset = verticesOffset;
			meshBuffer.indicesOffset = indicesOffset;

			verticesOffset += m_geometry[i].vertices.size();
			indicesOffset += m_geometry[i].indices.size();

			args.push_back(meshBuffer);
		}

		UINT numShaderRecords = args.size();
		UINT shaderRecordSize = shaderIdentifierSize + sizeof(args[0]);
		ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");
		for (auto& arg : args)
		{
			hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderIdentifier, shaderIdentifierSize, &arg, sizeof(MeshBuffer)));
		}

		m_hitGroupShaderTable = hitGroupShaderTable.GetResource();
	}
}

// Update frame-based values.
void D3D12RaytracingInOneWeekend::OnUpdate()
{
	m_timer.Tick();
	CalculateFrameStats();
	float elapsedTime = static_cast<float>(m_timer.GetElapsedSeconds());
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
	auto prevFrameIndex = m_deviceResources->GetPreviousFrameIndex();

	// Rotate the camera around Y axis.
	{
		float secondsToRotateAround = 60.0f;
		float angleToRotateBy = 360.0f * (elapsedTime / secondsToRotateAround);
		XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
		m_eye = XMVector3Transform(m_eye, rotate);
		m_up = XMVector3Transform(m_up, rotate);
		m_at = XMVector3Transform(m_at, rotate);
		UpdateCameraMatrices();
	}
}

void D3D12RaytracingInOneWeekend::DoRaytracing()
{
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	auto DispatchRays = [&](auto* commandList, auto* stateObject, auto* dispatchDesc)
		{
			// Since each shader table has only one shader record, the stride is same as the size.
			dispatchDesc->HitGroupTable.StartAddress = m_hitGroupShaderTable->GetGPUVirtualAddress();
			dispatchDesc->HitGroupTable.SizeInBytes = m_hitGroupShaderTable->GetDesc().Width;
			dispatchDesc->HitGroupTable.StrideInBytes = dispatchDesc->HitGroupTable.SizeInBytes / m_geometry.size();
			dispatchDesc->MissShaderTable.StartAddress = m_missShaderTable->GetGPUVirtualAddress();
			dispatchDesc->MissShaderTable.SizeInBytes = m_missShaderTable->GetDesc().Width;
			dispatchDesc->MissShaderTable.StrideInBytes = dispatchDesc->MissShaderTable.SizeInBytes;
			dispatchDesc->RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGPUVirtualAddress();
			dispatchDesc->RayGenerationShaderRecord.SizeInBytes = m_rayGenShaderTable->GetDesc().Width;
			dispatchDesc->Width = m_width;
			dispatchDesc->Height = m_height;
			dispatchDesc->Depth = 1;
			commandList->SetPipelineState1(stateObject);
			commandList->DispatchRays(dispatchDesc);
		};

	commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

	// Copy the updated scene constant buffer to GPU.
	memcpy(&m_mappedConstantData[frameIndex].constants, &m_frameCB[frameIndex], sizeof(m_frameCB[frameIndex]));
	auto cbGpuAddress = m_perFrameConstants->GetGPUVirtualAddress() + frameIndex * sizeof(m_mappedConstantData[0]);
	commandList->SetComputeRootConstantBufferView(GlobalRootSignatureParams::SceneConstantSlot, cbGpuAddress);

	// Bind the heaps, acceleration structure and dispatch rays.
	commandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
	// Set index and successive vertex buffer decriptor tables
	commandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::VertexBuffersSlot, m_indexBuffer.gpuDescriptorHandle);
	commandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::OutputViewSlot, m_raytracingOutputResourceUAVGpuDescriptor);
	commandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::AccelerationStructureSlot, m_topLevelAccelerationStructure->GetGPUVirtualAddress());

	D3D12_DISPATCH_RAYS_DESC dispatchDesc{ };
	DispatchRays(m_dxrCommandList.Get(), m_dxrStateObject.Get(), &dispatchDesc);
}

// Update the application state with the new resolution.
void D3D12RaytracingInOneWeekend::UpdateForSizeChange(UINT width, UINT height)
{
	DXSample::UpdateForSizeChange(width, height);
}

// Copy the raytracing output to the backbuffer.
void D3D12RaytracingInOneWeekend::CopyRaytracingOutputToBackbuffer()
{
	auto commandList = m_deviceResources->GetCommandList();
	auto renderTarget = m_deviceResources->GetRenderTarget();

	D3D12_RESOURCE_BARRIER preCopyBarriers[2];
	preCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
	preCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	commandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

	commandList->CopyResource(renderTarget, m_raytracingOutput.Get());

	D3D12_RESOURCE_BARRIER postCopyBarriers[2];
	postCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
	postCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	commandList->ResourceBarrier(ARRAYSIZE(postCopyBarriers), postCopyBarriers);
}

// Create resources that are dependent on the size of the main window.
void D3D12RaytracingInOneWeekend::CreateWindowSizeDependentResources()
{
	CreateRaytracingOutputResource();
	UpdateCameraMatrices();
}

// Release resources that are dependent on the size of the main window.
void D3D12RaytracingInOneWeekend::ReleaseWindowSizeDependentResources()
{
	m_raytracingOutput.Reset();
}

// Release all resources that depend on the device.
void D3D12RaytracingInOneWeekend::ReleaseDeviceDependentResources()
{
	m_raytracingGlobalRootSignature.Reset();
	m_raytracingLocalRootSignature.Reset();

	m_dxrDevice.Reset();
	m_dxrCommandList.Reset();
	m_dxrStateObject.Reset();

	m_descriptorHeap.Reset();
	m_descriptorsAllocated = 0;
	m_raytracingOutputResourceUAVDescriptorHeapIndex = UINT_MAX;
	m_perFrameConstants.Reset();
	m_rayGenShaderTable.Reset();
	m_missShaderTable.Reset();
	m_hitGroupShaderTable.Reset();
	m_indexBuffer.resource.Reset();
	m_vertexBuffer.resource.Reset();
	m_topLevelAccelerationStructure.Reset();

	for (auto& geometry : m_geometry)
	{
		geometry.m_bottomLevelAccelerationStructure.Reset();
	}
}

void D3D12RaytracingInOneWeekend::RecreateD3D()
{
	// Give GPU a chance to finish its execution in progress.
	try
	{
		m_deviceResources->WaitForGpu();
	}
	catch (HrException&)
	{
		// Do nothing, currently attached adapter is unresponsive.
	}
	m_deviceResources->HandleDeviceLost();
}

// Render the scene.
void D3D12RaytracingInOneWeekend::OnRender()
{
	if (!m_deviceResources->IsWindowVisible())
	{
		return;
	}

	m_deviceResources->Prepare();
	DoRaytracing();
	CopyRaytracingOutputToBackbuffer();

	m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT);
}

void D3D12RaytracingInOneWeekend::OnDestroy()
{
	// Let GPU finish before releasing D3D resources.
	m_deviceResources->WaitForGpu();
	OnDeviceLost();
}

// Release all device dependent resouces when a device is lost.
void D3D12RaytracingInOneWeekend::OnDeviceLost()
{
	ReleaseWindowSizeDependentResources();
	ReleaseDeviceDependentResources();
}

// Create all device dependent resources when a device is restored.
void D3D12RaytracingInOneWeekend::OnDeviceRestored()
{
	CreateDeviceDependentResources();
	CreateWindowSizeDependentResources();
}

// Compute the average frames per second and million rays per second.
void D3D12RaytracingInOneWeekend::CalculateFrameStats()
{
	static int frameCnt = 0;
	static double elapsedTime = 0.0f;
	double totalTime = m_timer.GetTotalSeconds();
	frameCnt++;

	// Compute averages over one second period.
	if ((totalTime - elapsedTime) >= 1.0f)
	{
		float diff = static_cast<float>(totalTime - elapsedTime);
		float fps = static_cast<float>(frameCnt) / diff; // Normalize to an exact second.

		frameCnt = 0;
		elapsedTime = totalTime;

		float MRaysPerSecond = (m_width * m_height * fps) / static_cast<float>(1e6);

		wstringstream windowText;
		windowText << setprecision(2) << fixed
			<< L"    fps: " << fps << L"     ~Million Primary Rays/s: " << MRaysPerSecond
			<< L"    GPU[" << m_deviceResources->GetAdapterID() << L"]: " << m_deviceResources->GetAdapterDescription();
		SetCustomWindowText(windowText.str().c_str());
	}
}

// Handle OnSizeChanged message event.
void D3D12RaytracingInOneWeekend::OnSizeChanged(UINT width, UINT height, bool minimized)
{
	if (!m_deviceResources->WindowSizeChanged(width, height, minimized))
	{
		return;
	}

	UpdateForSizeChange(width, height);

	ReleaseWindowSizeDependentResources();
	CreateWindowSizeDependentResources();
}

// Allocate a descriptor and return its index. 
// If the passed descriptorIndexToUse is valid, it will be used instead of allocating a new one.
UINT D3D12RaytracingInOneWeekend::AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse)
{
	auto descriptorHeapCpuBase = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
	if (descriptorIndexToUse >= m_descriptorHeap->GetDesc().NumDescriptors)
	{
		descriptorIndexToUse = m_descriptorsAllocated++;
	}
	*cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeapCpuBase, descriptorIndexToUse, m_descriptorSize);
	return descriptorIndexToUse;
}

// Create SRV for a buffer.
UINT D3D12RaytracingInOneWeekend::CreateBufferSRV(D3DBuffer* buffer, UINT numElements, UINT elementSize)
{
	auto device = m_deviceResources->GetD3DDevice();

	// SRV
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{ };
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Buffer.NumElements = numElements;
	if (elementSize == 0)
	{
		srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
		srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
		srvDesc.Buffer.StructureByteStride = 0;
	}
	else
	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		srvDesc.Buffer.StructureByteStride = elementSize;
	}
	UINT descriptorIndex = AllocateDescriptor(&buffer->cpuDescriptorHandle);
	device->CreateShaderResourceView(buffer->resource.Get(), &srvDesc, buffer->cpuDescriptorHandle);
	buffer->gpuDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), descriptorIndex, m_descriptorSize);
	return descriptorIndex;
}