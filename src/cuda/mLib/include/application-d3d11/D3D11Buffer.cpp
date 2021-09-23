
namespace ml
{
	template <class T>
	void D3D11Buffer<T>::init(GraphicsDevice& g, const std::vector<T>& data) {
		m_graphics = &g.castD3D11();
		m_data = data;
		createGPU();
	}

	template <class T>
	void D3D11Buffer<T>::releaseGPU() {
		SAFE_RELEASE(m_buffer);
		SAFE_RELEASE(m_srv);
		SAFE_RELEASE(m_uav);
	}

	template <class T>
	void D3D11Buffer<T>::createGPU() {
		releaseGPU();

		if (m_data.size() == 0) return;

		auto &device = m_graphics->getDevice();
		auto &context = m_graphics->getContext();

		D3D11_BUFFER_DESC desc;
		desc.ByteWidth = (unsigned int)(sizeof(T)*m_data.size());
		//desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;
		desc.StructureByteStride = sizeof(T);
		desc.Usage = D3D11_USAGE_DEFAULT;	// read/write GPU
		//desc.Usage = D3D11_USAGE_DYNAMIC;	// read-only GPU, write CPU
		desc.BindFlags = 0;
		if (hasSRV()) desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
		if (hasUAV()) desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
		
		D3D11_SUBRESOURCE_DATA initialData;
		initialData.pSysMem = m_data.data();
		initialData.SysMemPitch = 0;
		initialData.SysMemSlicePitch = 0;

		D3D_VALIDATE(device.CreateBuffer(&desc, &initialData, &m_buffer));
		if (hasSRV()) {
			D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;

			if (std::is_same<T, float>::value) descSRV.Format = DXGI_FORMAT_R32_FLOAT;
			else if (std::is_same<T, vec4f>::value) descSRV.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			else throw MLIB_EXCEPTION("TOOD implement the format for template T...");
			
			descSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			//descSRV.Buffer.ElementOffset = 0;
			//descSRV.Buffer.ElementWidth = sizeof(T);
			descSRV.Buffer.FirstElement = 0;
			descSRV.Buffer.NumElements = (unsigned int)m_data.size();
			D3D_VALIDATE(device.CreateShaderResourceView(m_buffer, &descSRV, &m_srv));
		}
		if (hasUAV()) {
			MLIB_ASSERT_STR(false, "not implemented yet");
			D3D_VALIDATE(device.CreateUnorderedAccessView(m_buffer, nullptr, &m_uav));
		}
		
		//context.UpdateSubresource(m_texture, 0, nullptr, m_image.getData(), (UINT)m_image.getWidth() * sizeof(vec4uc), (UINT)m_image.getWidth() * (UINT)m_image.getHeight() * sizeof(vec4uc));
		//context.GenerateMips(m_srv);
	}

	template <class T>
	void D3D11Buffer<T>::bindSRV(unsigned int slot /* = 0 */) const
	{
		if (m_srv == nullptr) return;
		m_graphics->getContext().VSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().GSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().HSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().DSSetShaderResources(slot, 1, &m_srv);
		m_graphics->getContext().PSSetShaderResources(slot, 1, &m_srv);
	}

	template <class T>
	void D3D11Buffer<T>::unbindSRV(unsigned int slot /* = 0 */) const
	{
		if (m_srv == nullptr) return;
		ID3D11ShaderResourceView* const srvs[] = { nullptr };
		m_graphics->getContext().VSSetShaderResources(slot, 1, srvs);
		m_graphics->getContext().GSSetShaderResources(slot, 1, srvs);
		m_graphics->getContext().HSSetShaderResources(slot, 1, srvs);
		m_graphics->getContext().DSSetShaderResources(slot, 1, srvs);
		m_graphics->getContext().PSSetShaderResources(slot, 1, srvs);
	}
}