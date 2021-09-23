
#ifndef APPLICATION_D3D11_D3D11GRAPHICSDEVICE_H_
#define APPLICATION_D3D11_D3D11GRAPHICSDEVICE_H_

namespace ml {

class D3D11GraphicsDevice : public GraphicsDevice
{
public:
    D3D11GraphicsDevice()
    {
        m_type = GraphicsDeviceTypeD3D11;
        m_device = nullptr;
        m_context = nullptr;
        m_renderTargetView = nullptr;
        m_debug = nullptr;
        m_swapChain = nullptr;
        m_depthBuffer = nullptr;
        m_depthState = nullptr;
        m_rasterState = nullptr;
        m_depthStencilView = nullptr;
        m_captureBufferColor = nullptr;
		m_captureBufferDepth = nullptr;
        m_samplerState = nullptr;
		m_featureLevel = D3D_FEATURE_LEVEL_9_1;


		ZeroMemory(&m_swapChainDesc, sizeof(m_swapChainDesc));
		m_swapChainDesc.BufferCount = 1;
		m_swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		m_swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
		m_swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
		m_swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;		
		m_swapChainDesc.SampleDesc.Count = 1;
		m_swapChainDesc.SampleDesc.Quality = 0;
		m_swapChainDesc.Windowed = TRUE;
		//set in init
		//m_swapChainDesc.OutputWindow = window.handle();
		//m_swapChainDesc.BufferDesc.Width = m_width;
		//m_swapChainDesc.BufferDesc.Height = m_height;

		m_bExternallyCreated = false;
    }

    ~D3D11GraphicsDevice()
    {
		//if (m_assets.size()) {
		//	std::cout << __FUNCTION__ " : unreleased assets found" << std::endl;
		//	printAssets();
		//	throw MLIB_EXCEPTION("found unreleased assets");
		//}

		if (!m_bExternallyCreated) {
			SAFE_RELEASE(m_rasterState);
			SAFE_RELEASE(m_context);
			SAFE_RELEASE(m_renderTargetView);
			SAFE_RELEASE(m_swapChain);
			SAFE_RELEASE(m_device);
			SAFE_RELEASE(m_depthBuffer);
			SAFE_RELEASE(m_depthState);
			SAFE_RELEASE(m_depthStencilView);
			SAFE_RELEASE(m_captureBufferColor);
			SAFE_RELEASE(m_captureBufferDepth);
			SAFE_RELEASE(m_samplerState);

			if (m_debug)
			{
				//m_debug->ReportLiveDeviceObjects(D3D11_RLDO_SUMMARY);
			}
			SAFE_RELEASE(m_debug);
		}
    }
	
	
	void init(ID3D11Device* device, ID3D11DeviceContext* context, IDXGISwapChain* swapChain, ID3D11RenderTargetView* rtv, ID3D11DepthStencilView* dsv) {
		m_swapChain = swapChain;
		swapChain->GetDesc(&m_swapChainDesc);
		m_device = device;
		m_context = context;
		
		m_renderTargetView = rtv;
		m_depthStencilView = dsv;

		m_shaderManager.init(*this);
		registerDefaultShaders();
	
		m_width = m_swapChainDesc.BufferDesc.Width;
		m_height = m_swapChainDesc.BufferDesc.Height;

		m_bExternallyCreated = true;
	} 

    void init(const WindowWin32& window);

	void initWithoutWindow();

	void resize(const WindowWin32 &window);
    void renderBeginFrame();
    void renderEndFrame(bool vsync);

	//this is a) not really needed and doesn't work with move semantics because the asset pointers change
	////! registers an asset from the device
	//void registerAsset(GraphicsAsset* asset);
	////! unregisters an asset from the device
	//void unregisterAsset(GraphicsAsset* asset);
	////! lists all assets
	//void printAssets();

    void setCullMode(D3D11_CULL_MODE mode);
    void toggleCullMode();
    void toggleWireframe();

	//! clears the back buffer (color and depth)
	void clear(const vec4f &clearColor = vec4f(0, 0, 0, 0), float clearDepth = 1.0f);
    void bindRenderTarget();

    D3D11ShaderManager& getShaderManager()
	{
        return m_shaderManager;
    }

    ID3D11Device& getDevice()
    {
        return *m_device;
    }

    ID3D11DeviceContext& getContext()
    {
        return *m_context;
    }

	void setViewport(unsigned int width, unsigned int height, float minDepth, float maxDepth, float topLeftX, float topLeftY) 
	{
		m_viewportWidth = width;
		m_viewportHeight = height;
		D3D11_VIEWPORT viewport;
		viewport.Width = (float)width;
		viewport.Height = (float)height;
		viewport.MinDepth = minDepth;
		viewport.MaxDepth = maxDepth;
		viewport.TopLeftX = topLeftX;
		viewport.TopLeftY = topLeftY;
		m_context->RSSetViewports(1, &viewport);
	}


	unsigned int getWidth() const {
		return m_width;
	}

	unsigned int getHeight() const {
		return m_height;
	}

	//! maps from integer pixel coordinates to NDC space [-1;1]^2
	vec2f pixelToNDC(const vec2i& p) const {
		return pixelToNDC(p, m_viewportWidth, m_viewportHeight);
	}

	//! maps from integer pixel coordinates to NDC space [-1;1]^2
	static vec2f pixelToNDC(const vec2i& p, unsigned int width, unsigned int height) {
		return vec2f(
			2.0f*(float)p.x / ((float)width - 1.0f) - 1.0f,
			1.0f - 2.0f*(float)p.y / ((float)height - 1.0f));
	}

private:

	//! creates depth buffer, depth stencil view, rendertarget view, and sets the view port to the current width/height (called by init and resize)
	void createViews();

    void registerDefaultShaders();

    UINT m_width, m_height;
	UINT m_viewportWidth, m_viewportHeight;
    ID3D11Device* m_device;
    ID3D11DeviceContext* m_context;
    ID3D11RenderTargetView* m_renderTargetView;
    ID3D11Debug* m_debug;

    IDXGISwapChain* m_swapChain;
	DXGI_SWAP_CHAIN_DESC m_swapChainDesc;	//settings


    ID3D11RasterizerState* m_rasterState;
    D3D11_RASTERIZER_DESC m_rasterDesc;

    ID3D11Texture2D* m_depthBuffer;
    ID3D11DepthStencilState* m_depthState;
    ID3D11DepthStencilView* m_depthStencilView;

    ID3D11SamplerState* m_samplerState;

    ID3D11Texture2D* m_captureBufferColor;
	ID3D11Texture2D* m_captureBufferDepth;

    std::set<GraphicsAsset*> m_assets;

    D3D11ShaderManager m_shaderManager;

	D3D_FEATURE_LEVEL m_featureLevel;
	
	bool m_bExternallyCreated; 
protected:
	void captureBackBufferColorInternal(ColorImageR8G8B8A8& result);
	void captureBackBufferDepthInternal(DepthImage32& result);
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11GRAPHICSDEVICE_H_
