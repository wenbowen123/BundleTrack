#ifndef INCLUDE_CORE_GRAPHICS_COLORUTILS_H_
#define INCLUDE_CORE_GRAPHICS_COLORUTILS_H_

namespace ml {

class ColorUtils {
 public:
  /**
  * Converts an HSL color value to RGB. Conversion formula
  * adapted from http://en.wikipedia.org/wiki/HSL_color_space.
  * Assumes rgb and hsl values are in [0, 1].
  */
  template <typename RGB, typename T>
  static RGB hslToRgb(const T& hsl) {
    auto test = hsl[0];
    static_assert(static_cast<bool>(std::is_same<decltype(test), float>::value), "hslToRgb assumes float 0-1 range");

    // Helper for hsl to rgb conversion
    auto hue2rgb = [] (float p, float q, float t) {
      if (t < 0.0f) { t += 1.0f; }
      if (t > 1.0f) { t -= 1.0f; }
      if (t < 1.0f / 6.0f) { return p + (q - p) * 6.0f * t; }
      if (t < 1.0f / 2.0f) { return q; }
      if (t < 2.0f / 3.0f) { return p + (q - p) * (2.0f / 3.0f - t) * 6.0f; }
      return p;
    };
    RGB rgb;
    float h = hsl[0], s = hsl[1], l = hsl[2];
    if (s == 0) {
      rgb[0] = rgb[1] = rgb[2] = l;  // achromatic
    } else {
      float q = l < 0.5f ? l * (1 + s) : l + s - l * s;
      float p = 2.0f * l - q;
      rgb[0] = hue2rgb(p, q, h + 1.0f / 3);
      rgb[1] = hue2rgb(p, q, h);
      rgb[2] = hue2rgb(p, q, h - 1.0f / 3);
    }
    return rgb;
  }

  /**
  * Converts an RGB color value to HSL. Conversion formula
  * adapted from http://en.wikipedia.org/wiki/HSL_color_space.
  * Assumes rgb and hsl values are in [0, 1].
  */
  template <typename RGB, typename T>
  static RGB rgbToHsl(const T& rgb) {
    auto test = rgb[0];
    static_assert(static_cast<bool>(std::is_same<decltype(test), float>::value), "rgbToHsl assumes float 0-1 range");
    const float r = rgb[0], g = rgb[1], b = rgb[2];
    
    float max = ml::math::max(r, g, b), min = ml::math::min(r, g, b);;
    float h, s, l = (max + min) / 2;

    if (max == min) {
      h = s = 0; // achromatic
    } else {
      float d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      if (max == r) { h = (g - b) / d + (g < b ? 6 : 0); }
      else if (max == g) { h = (b - r) / d + 2; }
      else if (max == b) { h = (r - g) / d + 4; }
      h /= 6;
    }
    RGB hsl;  hsl[0] = h;  hsl[1] = s;  hsl[2] = l;
    return hsl;
  }

  // Return a nice simple id-based color based on cycling through hsv space
  template<typename RGBA>
  static RGBA colorById(const int id) {
    float h = std::fmod(-3.88f * id, 2.0f * math::PIf);
    if (h < 0) { h += 2.0f * math::PIf; }
    h /= 2.0f * math::PIf;

    float hsl[4] = {h, 0.6f + 0.4f * std::sin(0.42f * id), 0.5f};

    RGBA rgba = hslToRgb<RGBA>(hsl);
    rgba[3] = 1;
    return rgba;
  }

  template <typename RGBA, size_t N>
  static std::array<RGBA, N> colorArrayByIdSeq(const int start) {
    std::array<RGBA, N> out;
    for (int i = start, ii = 0; i < start + (int)N; i++, ii++) {
      out[ii] = colorById<RGBA>(i);
    }
    return out;
  }

  static inline vec4uc toColor8(const vec3f &v)
  {
      return vec4uc(util::boundToByte(v.x * 255.0f),
          util::boundToByte(v.y * 255.0f),
          util::boundToByte(v.z * 255.0f), 255);
  }

  static inline vec3f toColor32(const vec4uc &v)
  {
      return vec3f(v.getVec3()) / 255.0f;
  }

  // TODO: this is only necessary because our .resize() function on images doesn't work for ColorImageR8G8B8A8
  static inline ColorImageR8G8B8A8 resizeImage8(const ColorImageR8G8B8A8 &input, int newWidth, int newHeight)
  {
      ColorImageR32G32B32 image32(input.getWidth(), input.getHeight());

      for (const auto &p : image32)
          p.value = toColor32(input(p.x, p.y));
      image32.resize(newWidth, newHeight, true);

      ColorImageR8G8B8A8 result(newWidth, newHeight);

      for (const auto &p : result)
          p.value = toColor8(image32(p.x, p.y));

      return result;
  }

};

}  // namespace ml

#endif  // INCLUDE_CORE_GRAPHICS_COLORUTILS_H_
