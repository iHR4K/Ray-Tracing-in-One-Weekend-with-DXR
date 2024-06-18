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

#pragma once

#include "DXSample.h"
#include "StepTimer.h"
#include "RaytracingHlslCompat.h"

namespace GlobalRootSignatureParams
{
	enum Value
	{
		OutputViewSlot = 0,
		AccelerationStructureSlot,
		SceneConstantSlot,
		VertexBuffersSlot,
		Count
	};
}

namespace LocalRootSignatureParams
{
	enum Value
	{
		MeshBufferSlot = 0,
		Count
	};
}

class D3D12RaytracingInOneWeekend : public DXSample
{
public:
	D3D12RaytracingInOneWeekend(UINT width, UINT height, std::wstring name);

	// IDeviceNotify
	virtual void OnDeviceLost() override;
	virtual void OnDeviceRestored() override;

	// Messages
	virtual void OnInit();
	virtual void OnUpdate();
	virtual void OnRender();
	virtual void OnSizeChanged(UINT width, UINT height, bool minimized);
	virtual void OnDestroy();
	virtual IDXGISwapChain* GetSwapchain() { return m_deviceResources->GetSwapChain(); }

private:
	static const UINT FrameCount = 3;

	// We'll allocate space for several of these and they will need to be padded for alignment.
	static_assert(sizeof(FrameBuffer) < D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, "Checking the size here.");

	union AlignedSceneConstantBuffer
	{
		FrameBuffer constants;
		uint8_t alignmentPadding[D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT];
	};
	AlignedSceneConstantBuffer* m_mappedConstantData;
	ComPtr<ID3D12Resource>       m_perFrameConstants;

	// DirectX Raytracing (DXR) attributes
	ComPtr<ID3D12Device5> m_dxrDevice;
	ComPtr<ID3D12GraphicsCommandList5> m_dxrCommandList;
	ComPtr<ID3D12StateObject> m_dxrStateObject;

	// Root signatures
	ComPtr<ID3D12RootSignature> m_raytracingGlobalRootSignature;
	ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature;

	// Descriptors
	ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
	UINT m_descriptorsAllocated;
	UINT m_descriptorSize;

	// Raytracing scene
	FrameBuffer m_frameCB[FrameCount];

	// Geometry
	struct D3DBuffer
	{
		ComPtr<ID3D12Resource> resource;
		D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle;
		D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorHandle;
	};

	struct Geometry
	{
		std::vector<Vertex> vertices;
		std::vector<int> indices;
		XMFLOAT4 albedo = XMFLOAT4{ 0.0, 0.0, 0.0, 1.0 };
		int materialId = 0;

		XMMATRIX transform = XMMatrixIdentity();

		size_t indicesOffsetInBytes = 0;
		size_t verticesOffsetInBytes = 0;


		// Acceleration structure
		ComPtr<ID3D12Resource> m_bottomLevelAccelerationStructure;
	};

	ComPtr<ID3D12Resource> m_topLevelAccelerationStructure;

	std::vector<Geometry> m_geometry;

	D3DBuffer m_vertexBuffer;
	D3DBuffer m_indexBuffer;

	// Raytracing output
	ComPtr<ID3D12Resource> m_raytracingOutput;
	D3D12_GPU_DESCRIPTOR_HANDLE m_raytracingOutputResourceUAVGpuDescriptor;
	UINT m_raytracingOutputResourceUAVDescriptorHeapIndex;

	// Shader tables
	static const wchar_t* c_hitGroupName;
	static const wchar_t* c_raygenShaderName;
	static const wchar_t* c_closestHitShaderName;
	static const wchar_t* c_missShaderName;
	ComPtr<ID3D12Resource> m_missShaderTable;
	ComPtr<ID3D12Resource> m_hitGroupShaderTable;
	ComPtr<ID3D12Resource> m_rayGenShaderTable;

	// Application state
	StepTimer m_timer;
	float m_curRotationAngleRad;
	XMVECTOR m_eye;
	XMVECTOR m_at;
	XMVECTOR m_up;

	void UpdateCameraMatrices();
	void InitializeScene();
	void RecreateD3D();
	void DoRaytracing();
	void CreateConstantBuffers();
	void CreateDeviceDependentResources();
	void CreateWindowSizeDependentResources();
	void ReleaseDeviceDependentResources();
	void ReleaseWindowSizeDependentResources();
	void CreateRaytracingInterfaces();
	void SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig);
	void CreateRootSignatures();
	void CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
	void CreateRaytracingPipelineStateObject();
	void CreateDescriptorHeap();
	void CreateRaytracingOutputResource();
	void BuildGeometry();
	void BuildAccelerationStructures();
	void BuildShaderTables();
	void UpdateForSizeChange(UINT clientWidth, UINT clientHeight);
	void CopyRaytracingOutputToBackbuffer();
	void CalculateFrameStats();
	UINT AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse = UINT_MAX);
	UINT CreateBufferSRV(D3DBuffer* buffer, UINT numElements, UINT elementSize);
};
