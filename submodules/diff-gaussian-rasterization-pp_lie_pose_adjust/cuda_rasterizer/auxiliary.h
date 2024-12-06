/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float dtheta_alpha(const float* dR, const float3 x)
{	
	float dta = dR[0] * x.x + dR[3] * x.y + dR[6] * x.z;
	return dta;
}

__forceinline__ __device__ float dtheta_beta(const float* dR, const float3 x)
{
	float dta = dR[1] * x.x + dR[4] * x.y + dR[7] * x.z;
	return dta;
}

__forceinline__ __device__ float dtheta_gamma(const float* dR, const float3 x)
{
	float dta = dR[2] * x.x + dR[5] * x.y + dR[8] * x.z;
	return dta;
}

__forceinline__ __device__ void get_dRs(const float* viewqtvec, float* Rr, float* Rx, float* Ry, float* Rz)
{
	float qr = viewqtvec[0];
	float qx = viewqtvec[1];
	float qy = viewqtvec[2];
	float qz = viewqtvec[3];

	Rr[0] = 0.f;
	Rr[1] = 2.f * qz;
	Rr[2] = - 2.f * qy;
	Rr[3] = - 2.f * qz;
	Rr[4] = 0.f;
	Rr[5] = 2.f * qx;
	Rr[6] = 2.f * qy;
	Rr[7] = - 2.f * qx;
	Rr[8] = 0.f;

	Rx[0] = 0.f;
	Rx[1] = 2.f * qy;
	Rx[2] = 2.f * qz;
	Rx[3] = 2.f * qy;
	Rx[4] = - 4.f * qx;
	Rx[5] = 2.f * qr;
	Rx[6] = 2.f * qz;
	Rx[7] = - 2.f * qr;
	Rx[8] = - 4.f * qx;

	Ry[0] = - 4.f * qy;
	Ry[1] = 2.f * qx;
	Ry[2] = - 2.f * qr;
	Ry[3] = 2.f * qx;
	Ry[4] = 0.f;
	Ry[5] = 2.f * qz;
	Ry[6] = 2.f * qr;
	Ry[7] = 2.f * qz;
	Ry[8] = - 4.f * qy;

	Rz[0] = - 4.f * qz;
	Rz[1] = 2.f * qr;
	Rz[2] = 2.f * qx;
	Rz[3] = - 2.f * qr;
	Rz[4] = - 4.f * qz;
	Rz[5] = 2.f * qy;
	Rz[6] = 2.f * qx;
	Rz[7] = 2.f * qy;
	Rz[8] = 0.f;
}

__forceinline__ __device__ void quat2rot(const float* viewqtvec, float* viewmatrix)
{
	viewmatrix[0] = 1.f - 2.f * (viewqtvec[4] * viewqtvec[4] + viewqtvec[5] * viewqtvec[5]);
	viewmatrix[1] = 2.f * (viewqtvec[3] * viewqtvec[4] + viewqtvec[6] * viewqtvec[5]);
	viewmatrix[2] = 2.f * (viewqtvec[3] * viewqtvec[5] - viewqtvec[6] * viewqtvec[4]);
	viewmatrix[3] = 0.f;
	viewmatrix[4] = 2.f * (viewqtvec[3] * viewqtvec[4] - viewqtvec[6] * viewqtvec[5]);
	viewmatrix[5] = 1.f - 2.f * (viewqtvec[3] * viewqtvec[3] + viewqtvec[5] * viewqtvec[5]);
	viewmatrix[6] = 2.f * (viewqtvec[4] * viewqtvec[5] + viewqtvec[6] * viewqtvec[3]);
	viewmatrix[7] = 0.f;
	viewmatrix[8] = 2.f * (viewqtvec[3] * viewqtvec[5] + viewqtvec[6] * viewqtvec[4]); 
	viewmatrix[9] = 2.f * (viewqtvec[4] * viewqtvec[5] - viewqtvec[6] * viewqtvec[3]);
	viewmatrix[10] = 1.f - 2.f * (viewqtvec[3] * viewqtvec[3] + viewqtvec[4] * viewqtvec[4]);
	viewmatrix[11] = 0.f;
	viewmatrix[12] = viewqtvec[0]; 
	viewmatrix[13] = viewqtvec[1]; 
	viewmatrix[14] = viewqtvec[2];
	viewmatrix[15] = 1.f;
}

__forceinline__ __device__ void projectView(const float* projmatrix, const float* viewmatrix, float* projview)
{
	projview[0] = (projmatrix[0] * viewmatrix[0] + projmatrix[8] * viewmatrix[2]);
	projview[1] = (projmatrix[5] * viewmatrix[1] + projmatrix[9] * viewmatrix[2]);
	projview[2] = (projmatrix[10] * viewmatrix[2]);
	projview[3] = (projmatrix[11] * viewmatrix[2]);
	projview[4] = (projmatrix[0] * viewmatrix[4] + projmatrix[8] * viewmatrix[6]);
	projview[5] = (projmatrix[5] * viewmatrix[5] + projmatrix[9] * viewmatrix[6]);
	projview[6] = (projmatrix[10] * viewmatrix[6]);
	projview[7] = (projmatrix[11] * viewmatrix[6]);
	projview[8] = (projmatrix[0] * viewmatrix[8] + projmatrix[8] * viewmatrix[10]);
	projview[9] = (projmatrix[5] * viewmatrix[9] + projmatrix[9] * viewmatrix[10]);
	projview[10] = (projmatrix[10] * viewmatrix[10]);
	projview[11] = (projmatrix[11] * viewmatrix[10]);
	projview[12] = (projmatrix[0] * viewmatrix[12] + projmatrix[8] * viewmatrix[14]);
	projview[13] = (projmatrix[5] * viewmatrix[13] + projmatrix[9] * viewmatrix[14]);
	projview[14] = (projmatrix[10] * viewmatrix[14] + projmatrix[14]);
	projview[15] = (projmatrix[11] * viewmatrix[14]);
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}	

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec3x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[3] * p.x + matrix[4] * p.y + matrix[5] * p.z,
		matrix[6] * p.x + matrix[7] * p.y + matrix[8] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec3x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[3] * p.y + matrix[6] * p.z,
		matrix[1] * p.x + matrix[4] * p.y + matrix[7] * p.z,
		matrix[2] * p.x + matrix[5] * p.y + matrix[8] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 inverseTranslationfromT(const float* T)
{	
	float3 d_inv = {
		-(T[0] * T[12] + T[1] * T[13] + T[2] * T[14]),
		-(T[4] * T[12] + T[5] * T[13] + T[6] * T[14]),
		-(T[8] * T[12] + T[9] * T[13] + T[10] * T[14]),
	};
	return d_inv;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif