#pragma once

#include "cuda_SimpleMatrixUtil.h"
#include "mLib.h"

namespace MatrixConversion
{
	static ml::mat4f toMlib(const float4x4& m) {
		return ml::mat4f(m.ptr());
	}
	static ml::mat3f toMlib(const float3x3& m) {
		return ml::mat3f(m.ptr());
	}
	static ml::vec4f toMlib(const float4& v) {
		return ml::vec4f(v.x, v.y, v.z, v.w);
	}
	static ml::vec3f toMlib(const float3& v) {
		return ml::vec3f(v.x, v.y, v.z);
	}
	static ml::vec4i toMlib(const int4& v) {
		return ml::vec4i(v.x, v.y, v.z, v.w);
	}
	static ml::vec3i toMlib(const int3& v) {
		return ml::vec3i(v.x, v.y, v.z);
	}
	static float4x4 toCUDA(const ml::mat4f& m) {
		return float4x4(m.getData());
	}
	static float3x3 toCUDA(const ml::mat3f& m) {
		return float3x3(m.getData());
	}

	static float4 toCUDA(const ml::vec4f& v) {
		return make_float4(v.x, v.y, v.z, v.w);
	}
	static float3 toCUDA(const ml::vec3f& v) {
		return make_float3(v.x, v.y, v.z);
	}
	static int4 toCUDA(const ml::vec4i& v) {
		return make_int4(v.x, v.y, v.z, v.w);
	}
	static int3 toCUDA(const ml::vec3i& v) {
		return make_int3(v.x, v.y, v.z);
	}

}
