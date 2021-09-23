
#ifndef APPLICATION_BASE_GRAPHICSASSET_H_
#define APPLICATION_BASE_GRAPHICSASSET_H_

namespace ml {

class GraphicsAsset
{
public:
	//! returns the asset name
	virtual std::string getName() const {
		return typeid(*this).name();
	}
	
protected:
	//! releases all GPU parts
	virtual void releaseGPU() = 0;
	//! (re-)creates all GPU parts
	virtual void createGPU() = 0;
};

}  // namespace ml

#endif  // APPLICATION_BASE_GRAPHICSASSET_H_