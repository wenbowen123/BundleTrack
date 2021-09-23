
namespace ml
{

	template<class T>
	void D3D11Texture3D<T>::init(GraphicsDevice &g, const Grid3<T>& data) {
		m_graphics = &g.castD3D11();
		m_data = data;
		createGPU();
	}

	template<class T>
	void D3D11Texture3D<T>::releaseGPU() {
		SAFE_RELEASE(m_texture);
		SAFE_RELEASE(m_srv);
	}

	template<class T>
	void D3D11Texture3D<T>::createGPU() {
		releaseGPU();

		if (m_data.getNumElements() == 0) return;

		auto &device = m_graphics->getDevice();
		auto &context = m_graphics->getContext();

		D3D11_TEXTURE3D_DESC desc;
		desc.Width = (UINT)m_data.getDimX();
		desc.Height = (UINT)m_data.getDimY();
		desc.Depth = (UINT)m_data.getDimZ();
		desc.MipLevels = 0;
		//desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;		//is set below baed on template
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

		if (std::is_same<T, float>::value) desc.Format = DXGI_FORMAT_R32_FLOAT;
		else if (std::is_same<T, vec4uc>::value) desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		else if (std::is_same<T, vec4f>::value) desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		else throw MLIB_EXCEPTION("TOOD implement the format for template T...");

		D3D_VALIDATE(device.CreateTexture3D(&desc, nullptr, &m_texture));
		D3D_VALIDATE(device.CreateShaderResourceView(m_texture, nullptr, &m_srv));

		context.UpdateSubresource(m_texture, 0, nullptr, m_data.getData(), (UINT)m_data.getDimX() * sizeof(RGBColor), (UINT)m_data.getDimX() * (UINT)m_data.getDimY() * sizeof(RGBColor));

		context.GenerateMips(m_srv);
	}

	template<class T>
	void D3D11Texture3D<T>::bind(unsigned int slot /* = 0 */) const {
		if (m_srv == nullptr) return;
		m_graphics->getContext().VSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().GSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().HSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().DSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().PSSetShaderResources(slot, 1, &m_srv);
	}

	template<class T>
	void D3D11Texture3D<T>::unbind(unsigned int slot /* = 0 */) const {
		ID3D11ShaderResourceView* srvNULL = nullptr;
		m_graphics->getContext().VSSetShaderResources(slot, 1, &srvNULL);
		m_graphics->getContext().GSSetShaderResources(slot, 1, &srvNULL);
		m_graphics->getContext().HSSetShaderResources(slot, 1, &srvNULL);
		m_graphics->getContext().DSSetShaderResources(slot, 1, &srvNULL);
		m_graphics->getContext().PSSetShaderResources(slot, 1, &srvNULL);
	}

}