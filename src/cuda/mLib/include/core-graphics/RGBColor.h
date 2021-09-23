#ifndef CORE_GRAPHICS_RGBCOLOR_H_
#define CORE_GRAPHICS_RGBCOLOR_H_

namespace ml
{

struct RGBColor : public vec4uc
{
    RGBColor() {}
    RGBColor(BYTE _r, BYTE _g, BYTE _b)
	{
		r = _r;
		g = _g;
		b = _b;
		a = 255;
	}
    RGBColor(BYTE _r, BYTE _g, BYTE _b, BYTE _a)
	{
		r = _r;
		g = _g;
		b = _b;
		a = _a;
	}
	
	RGBColor(const std::string &hex);
	RGBColor(const vec4uc& v) {	//this one should not be explicit as it's the same class
		r = v.r;
		g = v.g;
		b = v.b;
		a = v.a;
	}
	explicit RGBColor(const vec3uc& v) {
		r = v.r;
		g = v.g;
		b = v.b;
		a = 255;
	}
    explicit RGBColor(const vec3f &v);
    explicit RGBColor(const vec4f &v);

    RGBColor flipBlueAndRed() const
    {
        return RGBColor(b, g, r, a);
    }

    RGBColor grayscale() const
    {
		MLIB_ASSERT_STR(false, "this function is actually wrong: a) it should return a float b) it should be like: return (0.2126f  c.x + 0.7152f  c.y + 0.0722f * c.z) / 255.f; ");
        BYTE avg = BYTE(((int)r + (int)g + (int)b) / 3);
        return RGBColor(avg, avg, avg, 255);
    }

    RGBColor inverse() const
    {
        return RGBColor(255 - r, 255 - g, 255 - b, 255 - a);
    }

	static UINT distL1(RGBColor a, RGBColor b)
	{
		return std::abs(int(a.r) - int(b.r)) +
			std::abs(int(a.g) - int(b.g)) +
			std::abs(int(a.b) - int(b.b));
	}

	static UINT distL2(RGBColor a, RGBColor b)
	{
		int DiffR = int(a.r) - int(b.r);
		int DiffG = int(a.g) - int(b.g);
		int DiffB = int(a.b) - int(b.b);
		return DiffR * DiffR + DiffG * DiffG + DiffB * DiffB;
	}

	static RGBColor randomColor()
	{
		return RGBColor(rand() & 255, rand() & 255, rand() & 255);
	}

    static RGBColor interpolate(RGBColor LowColor, RGBColor HighColor, float s);

	operator vec3f() const
	{
		const float scale = 1.0f / 255.0f;
		return vec3f(r * scale, g * scale, b * scale);
	}
    operator vec4f() const
    {
        const float scale = 1.0f / 255.0f;
        return vec4f(r * scale, g * scale, b * scale, a * scale);
    }

	vec3f toVec3f() const {
		const float scale = 1.0f / 255.0f;
		return vec3f(r * scale, g * scale, b * scale);
	}

	vec4f toVec4f() const {
		const float scale = 1.0f / 255.0f;
		return vec4f(r * scale, g * scale, b * scale, a * scale);
	}


	static RGBColor colorPalette(unsigned int idx, unsigned int maxNum = 64) {
		//TODO make this a more reasonable function: the core idea is with the same 'maxNum' the palette is exactly the same; so need a deterministic procedural generator


		vec3uc colors[] = { //http://godsnotwheregodsnot.blogspot.ru/2012/09/color-distribution-methodology.html
			vec3uc(1, 0, 103),
			vec3uc(213, 255, 0),
			vec3uc(255, 0, 86),
			vec3uc(158, 0, 142),
			vec3uc(14, 76, 161),
			vec3uc(255, 229, 2),
			vec3uc(0, 95, 57),
			vec3uc(0, 255, 0),
			vec3uc(149, 0, 58),
			vec3uc(255, 147, 126),
			vec3uc(164, 36, 0),
			vec3uc(0, 21, 68),
			vec3uc(145, 208, 203),
			vec3uc(98, 14, 0),
			vec3uc(107, 104, 130),
			vec3uc(0, 0, 255),
			vec3uc(0, 125, 181),
			vec3uc(106, 130, 108),
			vec3uc(0, 174, 126),
			vec3uc(194, 140, 159),
			vec3uc(190, 153, 112),
			vec3uc(0, 143, 156),
			vec3uc(95, 173, 78),
			vec3uc(255, 0, 0),
			vec3uc(255, 0, 246),
			vec3uc(255, 2, 157),
			vec3uc(104, 61, 59),
			vec3uc(255, 116, 163),
			vec3uc(150, 138, 232),
			vec3uc(152, 255, 82),
			vec3uc(167, 87, 64),
			vec3uc(1, 255, 254),
			vec3uc(255, 238, 232),
			vec3uc(254, 137, 0),
			vec3uc(189, 198, 255),
			vec3uc(1, 208, 255),
			vec3uc(187, 136, 0),
			vec3uc(117, 68, 177),
			vec3uc(165, 255, 210),
			vec3uc(255, 166, 254),
			vec3uc(119, 77, 0),
			vec3uc(122, 71, 130),
			vec3uc(38, 52, 0),
			vec3uc(0, 71, 84),
			vec3uc(67, 0, 44),
			vec3uc(181, 0, 255),
			vec3uc(255, 177, 103),
			vec3uc(255, 219, 102),
			vec3uc(144, 251, 146),
			vec3uc(126, 45, 210),
			vec3uc(189, 211, 147),
			vec3uc(229, 111, 254),
			vec3uc(222, 255, 116),
			vec3uc(0, 255, 120),
			vec3uc(0, 155, 255),
			vec3uc(0, 100, 1),
			vec3uc(0, 118, 255),
			vec3uc(133, 169, 0),
			vec3uc(0, 185, 23),
			vec3uc(120, 130, 49),
			vec3uc(0, 255, 198),
			vec3uc(255, 110, 65),
			vec3uc(232, 94, 190)
		};

		const unsigned int p0 = 73856093;
		const unsigned int p1 = 19349669;

		const unsigned int _idx = (idx * p0 + p1) % 64;
		return RGBColor(colors[_idx]);

		/*
		const unsigned int p0 = 73856093;
		const unsigned int p1 = 19349669;
		idx = (idx * p0 + p1) % maxNum;

		float x = idx / (float)(maxNum - 1);
		float r = 0.0f;
		float g = 0.0f;
		float b = 1.0f;
		if (x >= 0.0f && x < 0.2f) {
			x = x / 0.2f;
			r = 0.0f;
			g = x;
			b = 1.0f;
		}
		else if (x >= 0.2f && x < 0.4f) {
			x = (x - 0.2f) / 0.2f;
			r = 0.0f;
			g = 1.0f;
			b = 1.0f - x;
		}
		else if (x >= 0.4f && x < 0.6f) {
			x = (x - 0.4f) / 0.2f;
			r = x;
			g = 1.0f;
			b = 0.0f;
		}
		else if (x >= 0.6f && x < 0.8f) {
			x = (x - 0.6f) / 0.2f;
			r = 1.0f;
			g = 1.0f - x;
			b = 0.0f;
		}
		else if (x >= 0.8f && x <= 1.0f) {
			x = (x - 0.8f) / 0.2f;
			r = 1.0f;
			g = 0.0f;
			b = x;
		}
		return RGBColor(vec3f(r, g, b));
		*/
	}

    static const RGBColor White;
    static const RGBColor Red;
    static const RGBColor Green;
    static const RGBColor Gray;
    static const RGBColor Blue;
    static const RGBColor Yellow;
    static const RGBColor Orange;
    static const RGBColor Magenta;
    static const RGBColor Black;
    static const RGBColor Cyan;
    static const RGBColor Purple;
};

//
// Exact comparison functions.  Does not match the alpha channel.
//
inline bool operator == (RGBColor left, RGBColor right)
{
    return ((left.r == right.r) && (left.g == right.g) && (left.b == right.b));
}

inline bool operator != (RGBColor left, RGBColor right)
{
    return ((left.r != right.r) || (left.g != right.g) || (left.b != right.b));
}

}  // namespace ml

#endif  // CORE_GRAPHICS_RGBCOLOR_H_
