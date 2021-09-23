
#ifndef APPLICATION_D3D11_D3D11TEXTURE3D_H_
#define APPLICATION_D3D11_D3D11TEXTURE3D_H_

namespace ml {

template<class T>
class D3D11Texture3D : public GraphicsAsset
{
public:
    D3D11Texture3D() {
        m_graphics = nullptr;
        m_texture = nullptr;
        m_srv = nullptr;
	}

	D3D11Texture3D(GraphicsDevice& g, const Grid3<T>& data) {
		m_texture = nullptr;
		m_srv = nullptr;
		init(g, data);
	}

	//! copy constructor
	D3D11Texture3D(const D3D11Texture3D& t) {
		m_texture = nullptr;
		m_srv = nullptr;
		init(g, t.getData());
	}
	//! move constructor
    D3D11Texture3D(D3D11Texture3D&& t) {
		m_graphics = nullptr;
		m_texture = nullptr;
		m_srv = nullptr;
		swap(*this, t);
    }

    ~D3D11Texture3D() {
		releaseGPU();
	}

	//! assignment operator
	void operator=(const D3D11Texture3D& t) {
		init(*t.m_graphics, t.getData());
	}

	//! move operator
	void operator=(D3D11Texture3D&& t) {
		swap(*this, t);
	}

	//! adl swap
	friend void swap(D3D11Texture3D& a, D3D11Texture3D& b) {
		std::swap(a.m_graphics, b.m_graphics);
		std::swap(a.m_data, b.m_data);
		std::swap(a.m_srv, b.m_srv);
		std::swap(a.m_texture, b.m_texture);
	}

    void init(GraphicsDevice &g, const Grid3<T>& data);

	void releaseGPU();
	void createGPU();

    void bind(unsigned int slot = 0) const;
	void unbind(unsigned int slot = 0) const;


    const Grid3<RGBColor>& getData() const {
        return m_data;
    }

private:
	D3D11GraphicsDevice* m_graphics;
    Grid3<RGBColor> m_data;
    ID3D11Texture3D* m_texture;
    ID3D11ShaderResourceView* m_srv;
};

}  // namespace ml

#include "D3D11Texture3D.cpp"

#endif  // APPLICATION_D3D11_D3D11TEXTURE3D_H_
