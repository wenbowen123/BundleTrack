
#ifndef _FREEIMAGEWRAPPER_HELPER_H_
#define _FREEIMAGEWRAPPER_HELPER_H_

namespace ml {

////////////////////////////////////////
// Conversions for free image warper ///
////////////////////////////////////////


//////////////////////
// Data Read Helper //
//////////////////////

//BYTE
template<class T>	inline void convertFromBYTE(T& output, const BYTE* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromBYTE<unsigned char>(unsigned char& output, const BYTE* input) {
	output = input[0];
}
template<>	inline void convertFromBYTE<vec3d>(vec3d& output, const BYTE* input) {
	output.z = input[0]/255.0;	
	output.y = input[0]/255.0;	
	output.x = input[0]/255.0;
}
template<>	inline void convertFromBYTE<vec4d>(vec4d& output, const BYTE* input) {
	output.z = input[0]/255.0;
	output.y = input[0]/255.0;
	output.x = input[0]/255.0;
	output.w = 1.0;
}
template<>	inline void convertFromBYTE<vec3f>(vec3f& output, const BYTE* input) {
	output.z = input[0]/255.0f;	
	output.y = input[0]/255.0f;	
	output.x = input[0]/255.0f;
}
template<>	inline void convertFromBYTE<vec4f>(vec4f& output, const BYTE* input) {
	output.z = input[0]/255.0f;
	output.y = input[0]/255.0f;
	output.x = input[0]/255.0f;
	output.w = 1.0f;
}
template<>	inline void convertFromBYTE<vec3i>(vec3i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[0];	
	output.x = input[0];
}
template<>	inline void convertFromBYTE<vec4i>(vec4i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[0];	
	output.x = input[0];
	output.w = 255;
}
template<>	inline void convertFromBYTE<vec3ui>(vec3ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[0];	
	output.x = input[0];
}
template<>	inline void convertFromBYTE<vec4ui>(vec4ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[0];	
	output.x = input[0];
	output.w = 255;
}
template<>	inline void convertFromBYTE<vec3uc>(vec3uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[0];	
	output.x = input[0];
}
template<>	inline void convertFromBYTE<vec4uc>(vec4uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[0];	
	output.x = input[0];
	output.w = 255;
}

//BYTE3
template<class T>	inline void convertFromBYTE3(T& output, const BYTE* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromBYTE3<vec3d>(vec3d& output, const BYTE* input) {
	output.z = input[0]/255.0;	
	output.y = input[1]/255.0;	
	output.x = input[2]/255.0;
}
template<>	inline void convertFromBYTE3<vec4d>(vec4d& output, const BYTE* input) {
	output.z = input[0]/255.0;
	output.y = input[1]/255.0;
	output.x = input[2]/255.0;
	output.w = 1.0;
}
template<>	inline void convertFromBYTE3<vec3f>(vec3f& output, const BYTE* input) {
	output.z = input[0]/255.0f;	
	output.y = input[1]/255.0f;	
	output.x = input[2]/255.0f;
}
template<>	inline void convertFromBYTE3<vec4f>(vec4f& output, const BYTE* input) {
	output.z = input[0]/255.0f;
	output.y = input[1]/255.0f;
	output.x = input[2]/255.0f;
	output.w = 1.0f;
}
template<>	inline void convertFromBYTE3<vec3i>(vec3i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	inline void convertFromBYTE3<vec4i>(vec4i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = 255;
}
template<>	inline void convertFromBYTE3<vec3ui>(vec3ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	inline void convertFromBYTE3<vec4ui>(vec4ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = 255;
}
template<>	inline void convertFromBYTE3<vec3uc>(vec3uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	inline void convertFromBYTE3<vec4uc>(vec4uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = 255;
}


//BYTE4
template<class T>	inline void convertFromBYTE4(T& output, const BYTE* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromBYTE4<vec3d>(vec3d& output, const BYTE* input) {
	output.z = input[0]/255.0;	
	output.y = input[1]/255.0;	
	output.x = input[2]/255.0;
}
template<>	inline void convertFromBYTE4<vec4d>(vec4d& output, const BYTE* input) {
	output.z = input[0]/255.0;
	output.y = input[1]/255.0;
	output.x = input[2]/255.0;
	output.w = input[3]/255.0;
}
template<>	inline void convertFromBYTE4<vec3f>(vec3f& output, const BYTE* input) {
	output.z = input[0]/255.0f;	
	output.y = input[1]/255.0f;	
	output.x = input[2]/255.0f;
}
template<>	inline void convertFromBYTE4<vec4f>(vec4f& output, const BYTE* input) {
	output.z = input[0]/255.0f;
	output.y = input[1]/255.0f;
	output.x = input[2]/255.0f;
	output.w = input[3]/255.0f;
}
template<>	inline void convertFromBYTE4<vec3i>(vec3i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	inline void convertFromBYTE4<vec4i>(vec4i& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = input[3];
}
template<>	inline void convertFromBYTE4<vec3ui>(vec3ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	inline void convertFromBYTE4<vec4ui>(vec4ui& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = input[3];
}
template<>	inline void convertFromBYTE4<vec3uc>(vec3uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
}
template<>	inline void convertFromBYTE4<vec4uc>(vec4uc& output, const BYTE* input) {
	output.z = input[0];	
	output.y = input[1];	
	output.x = input[2];
	output.w = input[3];
}


//USHORT
template<class T>	inline void convertFromUSHORT(T& output, const unsigned short* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromUSHORT<unsigned short>(unsigned short& output, const unsigned short* input) {
	output = *input;
}
template<>	inline void convertFromUSHORT<float>(float& output, const unsigned short* input) {
	output = (float)*input;
	output /= 1000.0f;
}
template<>	inline void convertFromUSHORT<double>(double& output, const unsigned short* input) {
	output = (double)*input;
	output /= 1000.0;
}

//USHORT3
template<class T>	inline void convertFromUSHORT3(T& output, const unsigned short* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromUSHORT3<vec3d>(vec3d& output, const unsigned short* input) {
	output.z = (double)input[0];
	output.y = (double)input[1];
	output.x = (double)input[2];
}
template<>	inline void convertFromUSHORT3<vec3f>(vec3f& output, const unsigned short* input) {
	output.z = (float)input[0];
	output.y = (float)input[1];
	output.x = (float)input[2];
}
template<>	inline void convertFromUSHORT3<vec3us>(vec3us& output, const unsigned short* input) {
	output.z = input[0];
	output.y = input[1];
	output.x = input[2];
}

template<>	inline void convertFromUSHORT3<unsigned short>(unsigned short& output, const unsigned short* input) {
	output = input[0];
}

//USHORT4
template<class T>	inline void convertFromUSHORT4(T& output, const unsigned short* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromUSHORT4<vec4d>(vec4d& output, const unsigned short* input) {
	output.z = (double)input[0];
	output.y = (double)input[1];
	output.x = (double)input[2];
	output.w = (double)input[3];
}
template<>	inline void convertFromUSHORT4<vec4us>(vec4us& output, const unsigned short* input) {
	output.z = input[0];
	output.y = input[1];
	output.x = input[2];
	output.w = input[3];
}
template<>	inline void convertFromUSHORT4<vec4uc>(vec4uc& output, const unsigned short* input) {
	//this is a hard-coded HDR to non-HDR conversion
	output.z = (unsigned char)(input[0] / 256);
	output.y = (unsigned char)(input[1] / 256);
	output.x = (unsigned char)(input[2] / 256);
	output.w = (unsigned char)(input[3] / 256);
}

template<>	inline void convertFromUSHORT4<vec3d>(vec3d& output, const unsigned short* input) {
	output.x = (double)input[0];
	output.y = (double)input[1];
	output.z = (double)input[2];
}
template<>	inline void convertFromUSHORT4<vec3us>(vec3us& output, const unsigned short* input) {
	output.z = input[0];
	output.y = input[1];
	output.x = input[2];
}

template<>	inline void convertFromUSHORT4<unsigned short>(unsigned short& output, const unsigned short* input) {
	output = input[0];
}

//FLOAT
template<class T>	inline void convertFromFLOAT(T& output, const float* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromFLOAT<float>(float& output, const float* input) {
	output = input[0];
}

//FLOAT3
template<class T>	inline void convertFromFLOAT3(T& output, const float* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromFLOAT3<vec3f>(vec3f& output, const float* input) {
	output.x = input[0];
	output.y = input[1];
	output.z = input[2];
}
template<>	inline void convertFromFLOAT3<vec4f>(vec4f& output, const float* input) {
	output.x = input[0];
	output.y = input[1];
	output.z = input[2];
	output.w = 1;
}

//FLOAT4
template<class T>	inline void convertFromFLOAT4(T& output, const float* input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertFromFLOAT4<vec4f>(vec4f& output, const float* input) {
	output.x = input[0];
	output.y = input[1];
	output.z = input[2];
	output.w = input[3];
}

///////////////////////
// DATA WRITE HELPER //
///////////////////////

//VEC3UC
template<class T>	inline void convertToVEC3UC(vec3uc& output, const T& input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertToVEC3UC<vec3d>(vec3uc& output, const vec3d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
}
template<>	inline void convertToVEC3UC<vec4d>(vec3uc& output, const vec4d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
}
template<>	inline void convertToVEC3UC<vec3f>(vec3uc& output, const vec3f& input) {
	output.x = (unsigned char)(input[0]*255.0f);
	output.y = (unsigned char)(input[1]*255.0f);
	output.z = (unsigned char)(input[2]*255.0f);
}
template<>	inline void convertToVEC3UC<vec4f>(vec3uc& output, const vec4f& input) {
	output.x = (unsigned char)(input[0]*255.0f);	
	output.y = (unsigned char)(input[1]*255.0f);	
	output.z = (unsigned char)(input[2]*255.0f);
}
template<>	inline void convertToVEC3UC<vec3i>(vec3uc& output, const vec3i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	inline void convertToVEC3UC<vec4i>(vec3uc& output, const vec4i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	inline void convertToVEC3UC<vec3ui>(vec3uc& output, const vec3ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	inline void convertToVEC3UC<vec4ui>(vec3uc& output, const vec4ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
}
template<>	inline void convertToVEC3UC<vec3uc>(vec3uc& output, const vec3uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
}
template<>	inline void convertToVEC3UC<vec4uc>(vec3uc& output, const vec4uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
}
template<>	inline void convertToVEC3UC<float>(vec3uc& output, const float& input) {
	convertToVEC3UC(output, vec3f(input));
}




//VEC4UC
template<class T>	inline void convertToVEC4UC(vec4uc& output, const T& input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertToVEC4UC<vec3d>(vec4uc& output, const vec3d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
	output.w = 255;
}
template<>	inline void convertToVEC4UC<vec4d>(vec4uc& output, const vec4d& input) {
	output.x = (unsigned char)(input[0]*255.0);	
	output.y = (unsigned char)(input[1]*255.0);	
	output.z = (unsigned char)(input[2]*255.0);
	output.w = (unsigned char)(input[3]*255.0);
}
template<>	inline void convertToVEC4UC<vec3f>(vec4uc& output, const vec3f& input) {
	output.x = (unsigned char)(input[0]*255.0f);
	output.y = (unsigned char)(input[1]*255.0f);
	output.z = (unsigned char)(input[2]*255.0f);
	output.w = 255;
}
template<>	inline void convertToVEC4UC<vec4f>(vec4uc& output, const vec4f& input) {
	output.x = (unsigned char)(input[0]*255.0);
	output.y = (unsigned char)(input[1]*255.0);
	output.z = (unsigned char)(input[2]*255.0);
	output.w = (unsigned char)(input[3]*255.0f);
}
template<>	inline void convertToVEC4UC<vec3i>(vec4uc& output, const vec3i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = 255;
}
template<>	inline void convertToVEC4UC<vec4i>(vec4uc& output, const vec4i& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = (unsigned char)input[3];
}
template<>	inline void convertToVEC4UC<vec3ui>(vec4uc& output, const vec3ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = 255;
}
template<>	inline void convertToVEC4UC<vec4ui>(vec4uc& output, const vec4ui& input) {
	output.x = (unsigned char)input[0];	
	output.y = (unsigned char)input[1];	
	output.z = (unsigned char)input[2];
	output.w = (unsigned char)input[3];
}
template<>	inline void convertToVEC4UC<vec3uc>(vec4uc& output, const vec3uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
	output.w = 255;
}
template<>	inline void convertToVEC4UC<vec4uc>(vec4uc& output, const vec4uc& input) {
	output.x = input[0];	
	output.y = input[1];	
	output.z = input[2];
	output.w = input[3];
}
template<>	inline void convertToVEC4UC<float>(vec4uc& output, const float& input) {
	convertToVEC4UC(output, vec4f(input));
}

//UCHAR
template<class T>	inline void convertToUCHAR(unsigned char& output, const T& input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertToUCHAR<unsigned char>(unsigned char& output, const unsigned char& input) {
	output = input;
}

//USHORT
template<class T>	inline void convertToUSHORT(unsigned short& output, const T& input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}

template<>	inline void convertToUSHORT<unsigned short>(unsigned short& output, const unsigned short& input) {
	output = input;
}
template<>	inline void convertToUSHORT<float>(unsigned short& output, const float& input) {
	output = (unsigned short)(input * 1000.0f);
}
template<>	inline void convertToUSHORT<double>(unsigned short& output, const double& input) {
	output = (unsigned short)(input * 1000.0);
}

//USHORT3
template<class T>	inline void convertToUSHORT3(vec3us& output, const T& input) {
	throw MLIB_EXCEPTION("Invalid Data Conversion");
	//static_assert(false, "Function should never be called");
}
template<>	inline void convertToUSHORT3<vec3us>(vec3us& output, const vec3us& input) {
	output.x = input[0];
	output.y = input[1];
	output.z = input[2];
}

} // namespace

#endif
