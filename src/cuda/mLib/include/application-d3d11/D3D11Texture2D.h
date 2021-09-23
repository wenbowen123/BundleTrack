
#ifndef APPLICATION_D3D11_D3D11TEXTURE2D_H_
#define APPLICATION_D3D11_D3D11TEXTURE2D_H_

namespace ml {

template<class T>
class D3D11Texture2D : public GraphicsAsset
{
public:
    D3D11Texture2D() {
        m_graphics = nullptr;
        m_texture = nullptr;
        m_srv = nullptr;
	}

	D3D11Texture2D(GraphicsDevice &g, const BaseImage<T>& image) {
		m_texture = nullptr;
		m_srv = nullptr;
		init(g, image);
	}

	//! copy constructor
	D3D11Texture2D(const D3D11Texture2D& t) {
		m_graphics = nullptr;
		m_texture = nullptr;
		m_srv = nullptr;
		init(*t.m_graphics, t.getImage());

	}

	//! move constructor
    D3D11Texture2D(D3D11Texture2D&& t) {
		m_graphics = nullptr;
		m_texture = nullptr;
		m_srv = nullptr;
		swap(*this, t);
    }

	~D3D11Texture2D() {
		releaseGPU();
	}

	//! assignment operator
	void operator=(const D3D11Texture2D& t) {
		init(*t.m_graphics, t.getImage());
	}
	//! move operator
    void operator=(D3D11Texture2D&& t) { 
		swap(*this, t);
    }

	//! adl swap
	friend void swap(D3D11Texture2D& a, D3D11Texture2D& b) {
		std::swap(a.m_graphics, b.m_graphics);
		std::swap(a.m_image, b.m_image);
		std::swap(a.m_srv, b.m_srv);
		std::swap(a.m_texture, b.m_texture);
	}

    void init(GraphicsDevice &g, const BaseImage<T>& image);

	void releaseGPU();
	void createGPU();

	void bind(unsigned int slot = 0) const;
	void unbind(unsigned int slot = 0) const;

	const BaseImage<T>& getImage() const   {
        return m_image;
    }

	bool isInit() const {
		return m_texture != nullptr;
	}

private:
	D3D11GraphicsDevice* m_graphics;
	BaseImage<T> m_image;
    ID3D11Texture2D* m_texture;
    ID3D11ShaderResourceView* m_srv;
};

}  // namespace ml

#include "D3D11Texture2D.cpp"

#endif  // APPLICATION_D3D11_D3D11TEXTURE2D_H_
