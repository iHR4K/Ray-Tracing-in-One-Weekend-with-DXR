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

#ifndef RAYTRACINGHLSLCOMPAT_H
#define RAYTRACINGHLSLCOMPAT_H

#ifdef HLSL
#include "HlslCompat.h"
#else
using namespace DirectX;

// Shader will use byte encoding to access indices.
typedef UINT16 Index;
#endif

struct FrameBuffer
{
	XMMATRIX projectionToWorld;
	XMMATRIX modelViewInverse;
	XMVECTOR cameraPosition;
};

struct MeshBuffer
{
	XMFLOAT4 albedo;
	int meshId;
	int materialId;
	int verticesOffset;
	int indicesOffset;
};

struct Vertex
{
	XMFLOAT3 position;
	float pad1;
	XMFLOAT3 normal;
	float pad2;
};

#endif // RAYTRACINGHLSLCOMPAT_H