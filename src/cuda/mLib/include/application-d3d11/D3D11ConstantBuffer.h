
#ifndef APPLICATION_D3D11_D3D11CONSTANTBUFFER_H_
#define APPLICATION_D3D11_D3D11CONSTANTBUFFER_H_

namespace ml {

template<class T>
class D3D11ConstantBuffer : public GraphicsAsset
{
public:
	D3D11ConstantBuffer() {
        m_graphics = nullptr;
		m_buffer = nullptr;
	}

	//! copy constructor
	D3D11ConstantBuffer(const D3D11ConstantBuffer& other) {
		m_graphics = nullptr;
		m_buffer = nullptr;
		init(*other.m_graphics);
		update(other.getData());
	}

	//! move constructor
	D3D11ConstantBuffer(D3D11ConstantBuffer&& other) {
		m_graphics = nullptr;
		m_buffer = nullptr;
		swap(*this, other);
	}

	~D3D11ConstantBuffer() {
		releaseGPU();
	}

	//! assignment operator
	void operator=(const D3D11ConstantBuffer& other) {
		init(*other.m_graphics);
		update(other.getData());
	}

	void operator=(D3D11ConstantBuffer&& other) {
		swap(*this, other);
	}

	//! adl swap
	friend void swap(D3D11ConstantBuffer& a, D3D11ConstantBuffer& b) {
		std::swap(a.m_graphics, b.m_graphics);
		std::swap(a.m_buffer, b.m_buffer);
	}

	void init(GraphicsDevice& g) {
        m_graphics = &g.castD3D11();
		createGPU();
	}

	void releaseGPU() {
		SAFE_RELEASE(m_buffer);
	}

	void createGPU() {
		releaseGPU();

		D3D11_BUFFER_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.ByteWidth = sizeof(T);
		desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		desc.CPUAccessFlags = 0;
		D3D_VALIDATE(m_graphics->getDevice().CreateBuffer( &desc, nullptr, &m_buffer ));
	}

    void updateAndBind(const T& data, unsigned int constantBufferIndex) {
        update(data);
		bind(constantBufferIndex);
    }

	void update(const T& data) {
		m_data = data;
        m_graphics->getContext().UpdateSubresource(m_buffer, 0, nullptr, &data, 0, 0);
	}

	void bind(unsigned int constantBufferIndex) {
		m_graphics->getContext().VSSetConstantBuffers(constantBufferIndex, 1, &m_buffer);
		m_graphics->getContext().GSSetConstantBuffers(constantBufferIndex, 1, &m_buffer);
		m_graphics->getContext().HSSetConstantBuffers(constantBufferIndex, 1, &m_buffer);
		m_graphics->getContext().DSSetConstantBuffers(constantBufferIndex, 1, &m_buffer);
		m_graphics->getContext().PSSetConstantBuffers(constantBufferIndex, 1, &m_buffer);
	}

	void unbind(unsigned int constantBufferIndex) {
		ID3D11Buffer* buffNULL[] = { nullptr };
		m_graphics->getContext()->VSSetConstantBuffers(constantBufferIndex, 1, buffNULL);
		m_graphics->getContext()->GSSetConstantBuffers(constantBufferIndex, 1, buffNULL);
		m_graphics->getContext()->HSSetConstantBuffers(constantBufferIndex, 1, buffNULL);
		m_graphics->getContext()->DSSetConstantBuffers(constantBufferIndex, 1, buffNULL);
		m_graphics->getContext()->PSSetConstantBuffers(constantBufferIndex, 1, buffNULL);
	}

	const T& getData() const {
		return m_data;
	}

private:
	D3D11GraphicsDevice* m_graphics;
	T m_data;
	ID3D11Buffer* m_buffer;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11CONSTANTBUFFER_H_
