
namespace ml
{

template<class T>
void D3D11Texture2D<T>::init(GraphicsDevice& g, const BaseImage<T>& image) {
    m_graphics = &g.castD3D11();
    m_image = image;
    createGPU();
}

template<class T>
void D3D11Texture2D<T>::releaseGPU() {
    SAFE_RELEASE(m_texture);
    SAFE_RELEASE(m_srv);
}

template<class T>
void D3D11Texture2D<T>::createGPU() {
    releaseGPU();

	if (m_image.getWidth() == 0 || m_image.getHeight() == 0) return;

    auto &device = m_graphics->getDevice();
	auto &context = m_graphics->getContext();

    D3D11_TEXTURE2D_DESC desc;
    desc.Width = (UINT)m_image.getWidth();
    desc.Height = (UINT)m_image.getHeight();
    desc.MipLevels = 0;
    desc.ArraySize = 1;
	//desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;		//is set below baed on template
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

	if (std::is_same<T, float>::value) desc.Format = DXGI_FORMAT_R32_FLOAT;
	else if (std::is_same<T, vec4uc>::value) desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	else if (std::is_same<T, vec4f>::value) desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	else throw MLIB_EXCEPTION("TOOD implement the format for template T...");

    D3D_VALIDATE(device.CreateTexture2D(&desc, nullptr, &m_texture));
    D3D_VALIDATE(device.CreateShaderResourceView(m_texture, nullptr, &m_srv));

    context.UpdateSubresource(m_texture, 0, nullptr, m_image.getData(), (UINT)m_image.getWidth() * sizeof(T), (UINT)m_image.getWidth() * (UINT)m_image.getHeight() * sizeof(T));

    context.GenerateMips(m_srv);
}

template<class T>
void D3D11Texture2D<T>::bind(unsigned int slot /* = 0 */) const {
    if (m_srv == nullptr) return;
	m_graphics->getContext().VSSetShaderResources(slot, 1, &m_srv);
	m_graphics->getContext().GSSetShaderResources(slot, 1, &m_srv);
	m_graphics->getContext().HSSetShaderResources(slot, 1, &m_srv);
	m_graphics->getContext().DSSetShaderResources(slot, 1, &m_srv);
	m_graphics->getContext().PSSetShaderResources(slot, 1, &m_srv);
}

template<class T>
void D3D11Texture2D<T>::unbind(unsigned int slot /*= 0*/) const {
	ID3D11ShaderResourceView* srvNULL = nullptr;
	m_graphics->getContext().VSSetShaderResources(slot, 1, &srvNULL);
	m_graphics->getContext().GSSetShaderResources(slot, 1, &srvNULL);
	m_graphics->getContext().HSSetShaderResources(slot, 1, &srvNULL);
	m_graphics->getContext().DSSetShaderResources(slot, 1, &srvNULL);
	m_graphics->getContext().PSSetShaderResources(slot, 1, &srvNULL);
}


}