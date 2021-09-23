#ifndef _FREEIMAGEWRAPPER_H_
#define _FREEIMAGEWRAPPER_H_

#include "freeImageWrapperHelper.h"

namespace ml {

	class FreeImageWrapper {
	public:

		static Image loadImage(const std::string& filename, bool debugPrint = false) {
			Image img;
			loadImage(filename, img, debugPrint);
			return img;
		}

		static void loadImage(const std::string& filename, Image& image, bool debugPrint = false) {

			//TODO get the image format from file
			Image::Format format = image.getFormat();

			switch (format) {
			case Image::FORMAT_ColorImageR8G8B8A8:
				loadImage(filename, *(ColorImageR8G8B8A8*)&image, debugPrint);
				break;
			case Image::FORMAT_ColorImageR32G32B32A32:
				loadImage(filename, *(ColorImageR32G32B32A32*)&image, debugPrint);
				break;
			case Image::FORMAT_ColorImageR32G32B32:
				loadImage(filename, *(ColorImageR32G32B32*)&image, debugPrint);
				break;
			case Image::FORMAT_DepthImage:
				loadImage(filename, *(DepthImage32*)&image, debugPrint);
				break;
			case Image::FORMAT_DepthImage16:
				loadImage(filename, *(DepthImage16*)&image, debugPrint);
				break;
			default:
				throw MLIB_EXCEPTION("unknown image format");
			}

			image.setFormat(format);
		}

		template<class T>
		static void loadImage(const std::string &filename, BaseImage<T>& resultImage, bool debugPrint = false) {
			if (util::getFileExtension(filename) == "mbinRGB" || util::getFileExtension(filename) == "mbindepth") {
				resultImage.loadFromBinaryMImage(filename);
				return;
			}

			FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
			FIBITMAP *dib(0);
			BYTE* bits;

			FreeImage_Initialise();
			fif = FreeImage_GetFileType(filename.c_str());
			dib = FreeImage_Load(fif, filename.c_str());
			if (!dib) throw MLIB_EXCEPTION("Could not load image: " + filename);
			FREE_IMAGE_TYPE fitype = FreeImage_GetImageType(dib);

			if (fitype != FIT_BITMAP && fitype != FIT_UINT16 && fitype != FIT_RGB16 && fitype != FIT_FLOAT && fitype != FIT_RGBAF && fitype != FIT_RGBF) 
				throw MLIB_EXCEPTION("Unknown image format");

			bits = FreeImage_GetBits(dib);
			unsigned int width = FreeImage_GetWidth(dib);
			unsigned int height = FreeImage_GetHeight(dib);
			unsigned int nBits = FreeImage_GetBPP(dib);
			unsigned int pitch = FreeImage_GetPitch(dib);

			resultImage.allocate(width, height);

			if (fitype == FIT_UINT16) {
				const BYTE* data = (BYTE*)bits;
				unsigned int bytesPerPixel = nBits / 8;
				MLIB_ASSERT(bytesPerPixel == 2);
				for (unsigned int y = 0; y < height; y++) {
					const BYTE* dataRowStart = data + (height - 1 - y)*pitch;
					for (unsigned int x = 0; x < width; x++) {
						convertFromUSHORT(resultImage(x, y), (USHORT*)&dataRowStart[x*bytesPerPixel]);
					}
				}
			}
			else if (fitype == FIT_BITMAP) {
				const BYTE* data = (BYTE*)bits;
				unsigned int bytesPerPixel = nBits / 8;
				if (bytesPerPixel == 1) {
					for (unsigned int y = 0; y < height; y++) {
						const BYTE* dataRowStart = data + (height - 1 - y)*pitch;
						for (unsigned int x = 0; x < width; x++) {
							convertFromBYTE(resultImage(x, y), &dataRowStart[x*bytesPerPixel]);
						}
					}
				}
				else if (bytesPerPixel == 3) {
					for (unsigned int y = 0; y < height; y++) {
						const BYTE* dataRowStart = data + (height - 1 - y)*pitch;
						for (unsigned int x = 0; x < width; x++) {
							convertFromBYTE3(resultImage(x, y), &dataRowStart[x*bytesPerPixel]);
						}
					}
				}
				else if (bytesPerPixel == 4) {
					for (unsigned int y = 0; y < height; y++) {
						const BYTE* dataRowStart = data + (height - 1 - y)*pitch;
						for (unsigned int x = 0; x < width; x++) {
							convertFromBYTE4(resultImage(x, y), &dataRowStart[x*bytesPerPixel]);
						}
					}
				}
				else {
					throw MLIB_EXCEPTION("Unknown image format");
				}
			}
			else if (fitype == FIT_RGB16) {
				const BYTE* data = (BYTE*)bits;
				unsigned int bytesPerPixel = nBits / 8;
				MLIB_ASSERT(bytesPerPixel == 6);
				for (unsigned int y = 0; y < height; y++) {
					const BYTE* dataRowStart = data + (height - 1 - y)*pitch;
					for (unsigned int x = 0; x < width; x++) {
						convertFromUSHORT3(resultImage(x, y), (USHORT*)&dataRowStart[x*bytesPerPixel]);
					}
				}
			} 
			else if (fitype == FIT_RGBA16) {
				//TODO test this part
				const BYTE* data = (BYTE*)bits;
				unsigned int bytesPerPixel = nBits / 8;
				MLIB_ASSERT(bytesPerPixel == 8);
				for (unsigned int y = 0; y < height; y++) {
					const BYTE* dataRowStart = data + (height - 1 - y)*pitch;
					for (unsigned int x = 0; x < width; x++) {
						convertFromUSHORT4(resultImage(x, y), (USHORT*)&dataRowStart[x*bytesPerPixel]);
					}
				}
			}
			else if (fitype == FIT_FLOAT) {
				float *data = (float*)bits;
				unsigned int bytesPerPixel = nBits / 8;
				MLIB_ASSERT(bytesPerPixel == 4);
				for (unsigned int y = 0; y < height; y++) {
					for (unsigned int x = 0; x < width; x++) {
						convertFromFLOAT(resultImage(x, height - 1 - y), &data[y * width + x]);
					}
				}
			}
			else if (fitype == FIT_RGBF) {
				float *data = (float *)bits;
				unsigned int bytesPerPixel = nBits / 8;
				MLIB_ASSERT(bytesPerPixel == 12);
				for (unsigned int y = 0; y < height; y++) {
					for (unsigned int x = 0; x < width; x++) {
						convertFromFLOAT3(resultImage(x, height - 1 - y), &data[3 * (y * width + x)]);
					}
				}
			}
			else if (fitype == FIT_RGBAF) {
				float *data = (float *)bits;
				unsigned int bytesPerPixel = nBits / 8;
				MLIB_ASSERT(bytesPerPixel == 16);
				for (unsigned int y = 0; y < height; y++) {
					for (unsigned int x = 0; x < width; x++) {
						convertFromFLOAT4(resultImage(x, height - 1 - y), &data[4 * (y * width + x)]);
					}
				}
			}

			FreeImage_Unload(dib);
			if (debugPrint) {
				std::cout << __FUNCTION__ << ":" << filename << " (width=" << width << ";height=" << height << "; " << resultImage.getNumChannels() << "; " << resultImage.getNumBytesPerChannel() << ")" << std::endl;
			}

			//
			// FreeImage_DeInitialise() is buggy and crashes any openMP code that uses loadImage
			//
			//FreeImage_DeInitialise();
		}



		static void saveImage(const std::string &filename, const Image& image, bool debugPrint = false) {

			Image::Format format = image.getFormat();

			switch (format) {
			case Image::FORMAT_ColorImageR8G8B8A8:
				saveImage(filename, *(const ColorImageR8G8B8A8*)&image, debugPrint);
				break;
			case Image::FORMAT_ColorImageR32G32B32A32:
				saveImage(filename, *(const ColorImageR32G32B32A32*)&image, debugPrint);
				break;
			case Image::FORMAT_ColorImageR32G32B32:
				saveImage(filename, *(const ColorImageR32G32B32*)&image, debugPrint);
				break;
			case Image::FORMAT_DepthImage:
				saveImage(filename, *(const DepthImage32*)&image, debugPrint);
				break;
			case Image::FORMAT_DepthImage16:
				saveImage(filename, *(const DepthImage16*)&image, debugPrint);
				break;
			default:
				throw MLIB_EXCEPTION("unknown image format");
			}

		}


		template<class T>
		static void saveImage(const std::string &filename, const BaseImage<T>& image, bool debugPrint = false) {
			if (util::getFileExtension(filename) == "mbinRGB" || util::getFileExtension(filename) == "mbindepth") {
				image.saveAsBinaryMImage(filename);
				return;
			}

			FreeImage_Initialise();

			unsigned int width = image.getWidth();
			unsigned int height = image.getHeight();

			const unsigned int bytesPerChannel = image.getNumBytesPerChannel();
			const unsigned int numChannels = image.getNumChannels();


			if (filename.length() > 4 && filename.find(".jpg") != std::string::npos ||
				filename.length() > 4 && filename.find(".png") != std::string::npos) {
				FREE_IMAGE_TYPE type = FIT_BITMAP;
				if (numChannels == 1 && bytesPerChannel == 2) type = FIT_UINT16;
				else if (numChannels == 3 && bytesPerChannel == 2) type = FIT_RGB16;
				FIBITMAP *dib = FreeImage_AllocateT(type, width, height, numChannels * 8);
				BYTE* bits = FreeImage_GetBits(dib);
				unsigned int pitch = FreeImage_GetPitch(dib);

				if (numChannels == 1 && bytesPerChannel == 1) {
					//unsigned char
					for (unsigned int y = 0; y < height; y++) {
						BYTE* bitsRowStart = bits + (height - 1 - y)*pitch;
						unsigned char* bitsRowStartUChar = (unsigned char*)bitsRowStart;
						for (unsigned int x = 0; x < width; x++) {
							unsigned char v;	convertToUCHAR(v, image(x, y));
							bitsRowStartUChar[x] = v;
						}
					}
				}
				else if (numChannels == 1 && bytesPerChannel == 2) {
					//depth map; unsigned short
					for (unsigned int y = 0; y < height; y++) {
						BYTE* bitsRowStart = bits + (height - 1 - y)*pitch;
						USHORT* bitsRowStartUShort = (USHORT*)bitsRowStart;
						for (unsigned int x = 0; x < width; x++) {
							USHORT v;	convertToUSHORT(v, image(x, y));
							bitsRowStartUShort[x] = v;
						}
					}
				}
				else if (numChannels == 1 && bytesPerChannel == 4) {
					//R32
					for (unsigned int y = 0; y < height; y++) {
						BYTE* bitsRowStart = bits + (height - 1 - y)*pitch;
						for (unsigned int x = 0; x < width; x++) {
							//TODO: a 32-bit float can't fit into a vec3uc
							vec3uc color;		convertToVEC3UC(color, image(x, y));
							bitsRowStart[x*numChannels + FI_RGBA_RED] = (unsigned char)color.x;
							bitsRowStart[x*numChannels + FI_RGBA_GREEN] = (unsigned char)color.y;
							bitsRowStart[x*numChannels + FI_RGBA_BLUE] = (unsigned char)color.z;
						}
					}
				}
				else if ((numChannels == 3 && bytesPerChannel == 1) || (numChannels == 3 && bytesPerChannel == 4)) {
					//color map; R8G8B8; R32G32B32
					for (unsigned int y = 0; y < height; y++) {
						BYTE* bitsRowStart = bits + (height - 1 - y)*pitch;
						for (unsigned int x = 0; x < width; x++) {
							vec3uc color;		convertToVEC3UC(color, image(x, y));
							bitsRowStart[x*numChannels + FI_RGBA_RED] = (unsigned char)color.x;
							bitsRowStart[x*numChannels + FI_RGBA_GREEN] = (unsigned char)color.y;
							bitsRowStart[x*numChannels + FI_RGBA_BLUE] = (unsigned char)color.z;
						}
					}
				}
				else if (numChannels == 3 && bytesPerChannel == 2) {
					//3x16bit 
					for (unsigned int y = 0; y < height; y++) {
						BYTE* bitsRowStart = bits + (height - 1 - y)*pitch;
						USHORT* bitsRowStartUShort = (USHORT*)bitsRowStart;
						for (unsigned int x = 0; x < width; x++) {
							vec3<unsigned short> v;		convertToUSHORT3(v, image(x, y));
							bitsRowStartUShort[x*numChannels + FI_RGBA_RED] = v.x;
							bitsRowStartUShort[x*numChannels + FI_RGBA_GREEN] = v.y;
							bitsRowStartUShort[x*numChannels + FI_RGBA_BLUE] = v.z;
						}
					}
				}
				else if ((numChannels == 4 && bytesPerChannel == 1) || (numChannels == 4 && bytesPerChannel == 4)) {
					if (filename.find(".jpg") != std::string::npos) throw MLIB_EXCEPTION("jpg does not support transparencies");
					//MLIB_ASSERT(filename.find(".jpg") == std::string::npos);	//jpgs with transparencies don't work...
					//color map; R8G8B8A8; R32G32B32A32
					for (unsigned int y = 0; y < height; y++) {
						BYTE* bitsRowStart = bits + (height - 1 - y)*pitch;
						for (unsigned int x = 0; x < width; x++) {
							vec4uc color;		convertToVEC4UC(color, image(x, y));
							bitsRowStart[x*numChannels + FI_RGBA_RED] = (unsigned char)color.x;
							bitsRowStart[x*numChannels + FI_RGBA_GREEN] = (unsigned char)color.y;
							bitsRowStart[x*numChannels + FI_RGBA_BLUE] = (unsigned char)color.z;
							bitsRowStart[x*numChannels + FI_RGBA_ALPHA] = (unsigned char)color.w;
						}
					}
				}
				else {
					throw MLIB_EXCEPTION("Unknown image format (" + std::to_string(image.getNumChannels()) + "|" + std::to_string(image.getNumBytesPerChannel()) + ")");
				}


				if (filename.length() > 4 && filename.find(".jpg") != std::string::npos) {
					FreeImage_Save(FIF_JPEG, dib, filename.c_str());
				}
				else if (filename.length() > 4 && filename.find(".png") != std::string::npos) {
					FreeImage_Save(FIF_PNG, dib, filename.c_str());
				}
				else {
					assert(false);
				}
				FreeImage_Unload(dib);
			}
			else {
				throw MLIB_EXCEPTION("Unknown file format");
			}

			if (debugPrint) {
				std::cout << __FUNCTION__ << ":" << filename << " (width=" << width << ";height=" << height << "; " << image.getNumChannels() << "; " << image.getNumBytesPerChannel() << ")" << std::endl;
			}
			FreeImage_DeInitialise();
		}
	private:

	};

} // end namespace

#endif
