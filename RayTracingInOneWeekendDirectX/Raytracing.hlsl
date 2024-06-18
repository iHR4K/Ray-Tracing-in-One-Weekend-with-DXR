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

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#define HLSL
#include "RaytracingHlslCompat.h"
#include "Random.hlsl"

float Schlick(const float cosine, const float refractionIndex)
{
    float r0 = (1 - refractionIndex) / (1 + refractionIndex);
    r0 *= r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}

UINT LAMBERTIAN = 0;

RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);
StructuredBuffer<int> Indices : register(t1, space0);
StructuredBuffer<Vertex> Vertices : register(t2, space0);

ConstantBuffer<FrameBuffer> g_frameCB : register(b0);
ConstantBuffer<MeshBuffer> meshBuffer : register(b1);

typedef BuiltInTriangleIntersectionAttributes MyAttributes;
struct RayPayload
{
    float4 colorAndDistance;
    float4 scatterDirection;
    uint seed;
};

// Retrieve hit world position.
float3 HitWorldPosition()
{
    return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float3 HitAttribute(float3 vertexAttribute[3], BuiltInTriangleIntersectionAttributes attr)
{
    return vertexAttribute[0] +
        attr.barycentrics.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.barycentrics.y * (vertexAttribute[2] - vertexAttribute[0]);
}

RayPayload ScatterLambertian(in float4 albedo, in float3 worldRayDirection, in float3 normal, in float t, in uint seed)
{
    bool isScattered = dot(worldRayDirection, normal) < 0;
    float4 colorAndDistance = float4(albedo.rgb, t);
    float4 scatter = float4(normal + RandomInUnitSphere(seed), isScattered ? 1 : 0);

    RayPayload payload = (RayPayload) 0;
    payload.colorAndDistance = colorAndDistance;
    payload.scatterDirection = scatter;
    payload.seed = seed;
    return payload;
}

RayPayload ScatterMetal(in float4 albedo, in float3 worldRayDirection, in float3 normal, in float t, in uint seed)
{
    const float3 reflected = reflect(worldRayDirection, normal);
    const bool isScattered = dot(reflected, normal) > 0;
    
    const float4 colorAndDistance = float4(albedo.rgb, t);
    const float4 scatter = float4(reflected + albedo.w * RandomInUnitSphere(seed), isScattered ? 1 : 0);
    
    RayPayload payload = (RayPayload) 0;
    payload.colorAndDistance = colorAndDistance;
    payload.scatterDirection = scatter;
    payload.seed = seed;
    return payload;
}

RayPayload ScatterDielectric(in float4 albedo, in float3 worldRayDirection, in float3 normal, in float t, in uint seed, float refractionIndex)
{
    //
    const float DoN = dot(worldRayDirection, normal);
    const float3 outwardNormal = DoN > 0 ? -normal : normal;
    const float niOverNt = DoN > 0 ? refractionIndex : 1 / refractionIndex;
    const float cosine = DoN > 0 ? refractionIndex * DoN : -DoN;

    const float3 refracted = refract(worldRayDirection, outwardNormal, niOverNt);
    const float reflectProb = refracted != float3(0, 0, 0) ? Schlick(cosine, refractionIndex) : 1;

    const float4 color = float4(1.0, 1.0, 1.0, 1.0);
	
    RayPayload payload = (RayPayload) 0;
    payload.colorAndDistance = float4(color.rgb, t);
    payload.scatterDirection = RandomFloat(seed) < reflectProb ? float4(reflect(worldRayDirection, normal), 1) : float4(refracted, 1);
    payload.seed = seed;
    return payload;
}

void InitArray(inout float3 array[], int count)
{
    for (int i = 0; i < count; ++i)
    {
        array[i] = float3(0, 0, 0);
    }
}

// Inspired with:
// https://github.com/GPSnoopy/RayTracingInVulkan
[shader("raygeneration")]
void MyRaygenShader()
{
    uint TotalNumberOfSamples = 4; // TODO: should be in camera
    uint randomSeed = InitRandomSeed(InitRandomSeed(DispatchRaysIndex().x, DispatchRaysIndex().y), TotalNumberOfSamples);
    uint pixelRandomSeed = 1; // TODO: pass with raypayload
    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    uint bounces = 8;
    float aperture = 0.1;
    float focusDistance = 10.4;
    float3 color = float3(0.0, 0.0, 0.0);
    
    for (int sample = 0; sample < TotalNumberOfSamples; sample++)
    {
        float3 rayColor = float3(1.0, 1.0, 1.0);
       
        const float2 pixel = float2(DispatchRaysIndex().x + RandomFloat(pixelRandomSeed), DispatchRaysIndex().y + RandomFloat(pixelRandomSeed));
        float2 uv = (pixel / DispatchRaysDimensions().xy) * 2.0 - 1.0;
        uv.y *= -1; // directx 
 
        float2 offset = aperture / 2 * RandomInUnitDisk(randomSeed);
        float4 origin = mul(float4(offset, 0, 1), g_frameCB.modelViewInverse);
        float4 target = mul((float4(uv.x, uv.y, 1, 1)), g_frameCB.projectionToWorld);
        float4 direction = mul(float4(normalize(target.xyz * focusDistance - float3(offset, 0)), 0), g_frameCB.modelViewInverse);
       
        for (int i = 0; i <= bounces; ++i)
        {
            if (i == bounces)
            {
                rayColor = float3(0.0, 0.0, 0.0);
                break;
            }
            
            RayDesc ray;
            ray.Origin = origin;
            ray.Direction = direction;
            ray.TMin = 0.001;
            ray.TMax = 10000.0;
        
            RayPayload payload = (RayPayload) 0;
            payload.seed = randomSeed;

            TraceRay(Scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, ray, payload);
            
            const float3 hitColor = payload.colorAndDistance.rgb;
            const float t = payload.colorAndDistance.w;
            const bool isScattered = payload.scatterDirection.w > 0;
   
            rayColor *= hitColor;
            
            if (t < 0 || !isScattered)
            {
                break;
            }
            
            origin = origin + t * direction;
            direction = float4(payload.scatterDirection.xyz, 0);
     
        }
        color += rayColor;
    }
    
    color /= TotalNumberOfSamples;
    color = sqrt(color);

    // Write the raytraced color to the output texture.
    RenderTarget[DispatchRaysIndex().xy].rgb = color;
    RenderTarget[DispatchRaysIndex().xy].a = 1.0;
}

[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
    float3 hitPosition = HitWorldPosition();
    float3 worlRayDirection = WorldRayDirection();
 
    uint indicesPerTriangle = 3;
    uint baseIdx = indicesPerTriangle * PrimitiveIndex();
    uint indexWithOffset = baseIdx + meshBuffer.indicesOffset;
    
    int i0 = Indices[indexWithOffset + 0];
    int i1 = Indices[indexWithOffset + 1];
    int i2 = Indices[indexWithOffset + 2];

    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 vertexNormals[3] =
    {
        Vertices[i0 + meshBuffer.verticesOffset].normal,
        Vertices[i1 + meshBuffer.verticesOffset].normal,
        Vertices[i2 + meshBuffer.verticesOffset].normal 
    };

    // Compute the triangle's normal.
    // This is redundant and done for illustration purposes 
    // as all the per-vertex normals are the same and match triangle's normal in this sample. 
    float3 triangleNormal = HitAttribute(vertexNormals, attr);
    float t = RayTCurrent();
   
    int materialId = meshBuffer.materialId;
    switch (materialId)
    {
        case 0:
            payload = ScatterLambertian(meshBuffer.albedo, worlRayDirection, triangleNormal, t, payload.seed);
            break;
        case 1:
            payload = ScatterMetal(meshBuffer.albedo, worlRayDirection, triangleNormal, t, payload.seed);
            break;
        case 2:
            float refractionIndex = meshBuffer.albedo.x;
            payload = ScatterDielectric(meshBuffer.albedo, worlRayDirection, triangleNormal, t, payload.seed, refractionIndex);
            break;
    }
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    float a = 0.5 * (normalize(WorldRayDirection().y + 1.0));
    payload.colorAndDistance.rgb = (1.0 - a) * float3(1.0, 1.0, 1.0) + a * float3(0.5, 0.7, 1.0);
    payload.colorAndDistance.w = -1;
}

#endif // RAYTRACING_HLSL