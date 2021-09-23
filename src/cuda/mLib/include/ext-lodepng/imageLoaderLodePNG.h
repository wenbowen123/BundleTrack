#ifndef _EXT_LODEPNG_IMAGELOADERLODEPNG_H_
#define _EXT_LODEPNG_IMAGELOADERLODEPNG_H_

namespace ml {

class LodePNG
{
public:
  static ColorImageR8G8B8A8 load(const std::string &filename);
	static void save(const ColorImageR8G8B8A8 &image, const std::string &filename, bool saveTransparency = false);
};

}  // namespace ml

#endif  // _EXT_LODEPNG_IMAGELOADERLODEPNG_H_
