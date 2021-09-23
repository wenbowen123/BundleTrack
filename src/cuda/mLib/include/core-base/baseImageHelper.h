#ifndef _BASEIMAGE_HELPER_H_
#define _BASEIMAGE_HELPER_H_

namespace ml {

namespace BaseImageHelper {

    inline vec3f convertHSVtoRGB(const vec3f& hsv) {
        float H = hsv[0];
        float S = hsv[1];
        float V = hsv[2];

        float hd = H / 60.0f;
        unsigned int h = (unsigned int)hd;
        float f = hd - h;

        float p = V*(1.0f - S);
        float q = V*(1.0f - S*f);
        float t = V*(1.0f - S*(1.0f - f));

        if (h == 0 || h == 6)
        {
            return vec3f(V, t, p);
        }
        else if (h == 1)
        {
            return vec3f(q, V, p);
        }
        else if (h == 2)
        {
            return vec3f(p, V, t);
        }
        else if (h == 3)
        {
            return vec3f(p, q, V);
        }
        else if (h == 4)
        {
            return vec3f(t, p, V);
        }
        else
        {
            return vec3f(V, p, q);
        }
    }

	//! untested!
	inline vec3f convertRGBtoHSV(const vec3f& rgb)
	{
		vec3f hsv;
		const float r = rgb.r;
		const float g = rgb.g;
		const float b = rgb.b;

		float min, max, delta;
		min = std::min(std::min(r, g), b);
		max = std::max(std::max(r, g), b);
		hsv[2] = max;				// v
		delta = max - min;
		if (max != 0)
			hsv[1] = delta / max;		// s
		else {
			// r = g = b = 0		// s = 0, v is undefined
			hsv[1] = 0;
			hsv[0] = -1; // undefined hue
			return hsv;
		}
		if (r == max)
			hsv[0] = (g - b) / delta;		// between yellow & magenta
		else if (g == max)
			hsv[0] = 2 + (b - r) / delta;	// between cyan & yellow
		else
			hsv[0] = 4 + (r - g) / delta;	// between magenta & cyan
		hsv[0] *= 60;				// degrees
		if (hsv[0] < 0)
			hsv[0] += 360;
		return hsv;
	}

	inline vec3f convertDepthToHSV(float depth, float depthMin = 0.0f, float depthMax = 1.0f) {
		float depthZeroOne = (depth - depthMin) / (depthMax - depthMin);
		float x = 1.0f - depthZeroOne;
		if (x < 0.0f)	x = 0.0f;
		if (x > 1.0f)	x = 1.0f;
		return vec3f(240.0f*x, 1.0f, 0.5f);
	}

    inline vec3f convertDepthToRGB(float depth, float depthMin = 0.0f, float depthMax = 1.0f) {
        float depthZeroOne = (depth - depthMin) / (depthMax - depthMin);
        float x = 1.0f - depthZeroOne;
        if (x < 0.0f)	x = 0.0f;
        if (x > 1.0f)	x = 1.0f;
        return convertHSVtoRGB(vec3f(240.0f*x, 1.0f, 0.5f));
    }

    inline vec4f convertDepthToRGBA(float depth, float depthMin = 0.0f, float depthMax = 1.0f) {
        vec3f d = convertDepthToRGB(depth, depthMin, depthMax);
        return vec4f(d, 1.0f);
    }

	template<class T, class S> 
	inline static void convertBaseImagePixel(T& out, const S& in);

	template<> inline void convertBaseImagePixel<vec3f, vec3uc>(vec3f& out, const vec3uc& in) {
		out.x = in.x / 255.0f;
		out.y = in.y / 255.0f;
		out.z = in.z / 255.0f;
	}

	template<> inline void convertBaseImagePixel<vec3uc, vec3f>(vec3uc& out, const vec3f& in) {
		out.x = (unsigned char)(in.x * 255);
		out.y = (unsigned char)(in.y * 255);
		out.z = (unsigned char)(in.z * 255);
	}

	template<> inline void convertBaseImagePixel<vec4f, vec4uc>(vec4f& out, const vec4uc& in) {
		out.x = in.x / 255.0f;
		out.y = in.y / 255.0f;
		out.z = in.z / 255.0f;
		out.w = in.w / 255.0f;
	}

	template<> inline void convertBaseImagePixel<vec4uc, vec4f>(vec4uc& out, const vec4f& in) {
		out.x = (unsigned char)(in.x * 255);
		out.y = (unsigned char)(in.y * 255);
		out.z = (unsigned char)(in.z * 255);
		out.w = (unsigned char)(in.w * 255);
	}

	template<> inline void convertBaseImagePixel<vec3f, float>(vec3f& out, const float& in) {
		out = convertDepthToRGB(in);
	}

	template<> inline void convertBaseImagePixel<vec3uc, float>(vec3uc& out, const float& in) {
		vec3f tmp = convertDepthToRGB(in);
		convertBaseImagePixel(out, tmp);
	}
	template<> inline void convertBaseImagePixel<vec4f, float>(vec4f& out, const float& in) {
		out = vec4f(convertDepthToRGB(in), 1.0f);
		out.w = 0.0f;
	}

	template<> inline void convertBaseImagePixel<vec4uc, float>(vec4uc& out, const float& in) {
		vec4f tmp(convertDepthToRGB(in));
		convertBaseImagePixel(out, tmp);
	}



	template<> inline void convertBaseImagePixel<vec3uc, vec4uc>(vec3uc& out, const vec4uc& in) {
		out.x = in.x;
		out.y = in.y;
		out.z = in.z;
	}

	template<> inline void convertBaseImagePixel<vec3f, vec4uc>(vec3f& out, const vec4uc& in) {
		out.x = in.x / 255.0f;
		out.y = in.y / 255.0f;
		out.z = in.z / 255.0f;
	}
	template<> inline void convertBaseImagePixel<vec4uc, vec3f>(vec4uc& out, const vec3f& in) {
		out.x = (unsigned char)(in.x * 255);
		out.y = (unsigned char)(in.y * 255);
		out.z = (unsigned char)(in.z * 255);
		out.w = (unsigned char)255;
	}
	template<> inline void convertBaseImagePixel<vec4uc, vec3uc>(vec4uc& out, const vec3uc& in) {
		out.x = in.x;
		out.y = in.y;
		out.z = in.z;
		out.w = (unsigned char)255;
	}
	template<> inline void convertBaseImagePixel<vec3f, vec4f>(vec3f& out, const vec4f& in) {
		out.x = in.x;
		out.y = in.y;
		out.z = in.z;
	}
	template<> inline void convertBaseImagePixel<vec4f, vec3f>(vec4f& out, const vec3f& in) {
		out.x = in.x;
		out.y = in.y;
		out.z = in.z;
		out.w = 1.0f;
	}

};

} // namespace ml

#endif

