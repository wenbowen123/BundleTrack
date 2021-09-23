
#ifndef APPLICATION_D3D11_D3D11RENDERTARGET_H_
#define APPLICATION_D3D11_D3D11RENDERTARGET_H_

namespace ml {

class D3D11RenderTarget : public GraphicsAsset
{
public:
	D3D11RenderTarget()	{
		m_graphics = nullptr;
		m_width = 0;
		m_height = 0;

		m_targets = nullptr;
		m_targetsRTV = nullptr;
		m_targetsSRV = nullptr;

		m_depthStencil = nullptr;
		m_depthStencilDSV = nullptr;
		m_depthStencilSRV = nullptr;

		m_captureTextures = nullptr;
		m_captureDepth = nullptr;

		m_bHasSRVs = false;
	}

	D3D11RenderTarget(GraphicsDevice &g, unsigned int width, unsigned int height, const std::vector<DXGI_FORMAT>& formats = std::vector < DXGI_FORMAT > {DXGI_FORMAT_R8G8B8A8_UNORM}, bool createSRVs = false) {
		m_graphics = nullptr;
		m_width = 0;
		m_height = 0;

		m_targets = nullptr;
		m_targetsRTV = nullptr;
		m_targetsSRV = nullptr;

		m_depthStencil = nullptr;
		m_depthStencilDSV = nullptr;
		m_depthStencilSRV = nullptr;

		m_captureTextures = nullptr;
		m_captureDepth = nullptr;

		m_bHasSRVs = false;

        init(g, width, height, formats, createSRVs);
    }

	//! copy constructor
	D3D11RenderTarget(const D3D11RenderTarget& other) {
		m_graphics = nullptr;
		m_width = 0;
		m_height = 0;

		m_targets = nullptr;
		m_targetsRTV = nullptr;
		m_targetsSRV = nullptr;

		m_depthStencil = nullptr;
		m_depthStencilDSV = nullptr;
		m_depthStencilSRV = nullptr;

		m_captureTextures = nullptr;
		m_captureDepth = nullptr;

		m_bHasSRVs = false;

		init(*other.m_graphics, other.m_width, other.m_height, other.m_textureFormats, other.m_bHasSRVs);
	}

	//! move constructor
	D3D11RenderTarget(D3D11RenderTarget&& other) {
		m_graphics = nullptr;
		m_width = 0;
		m_height = 0;

		m_targets = nullptr;
		m_targetsRTV = nullptr;
		m_targetsSRV = nullptr;

		m_depthStencil = nullptr;
		m_depthStencilDSV = nullptr;
		m_depthStencilSRV = nullptr;

		m_captureTextures = nullptr;
		m_captureDepth = nullptr;

		m_bHasSRVs = false;

		swap(*this, other);
	}


	~D3D11RenderTarget() {
		releaseGPU();
	}

	//! assignment operator
	void operator=(const D3D11RenderTarget& other) {
		init(*other.m_graphics, other.m_width, other.m_height, other.m_textureFormats, other.m_bHasSRVs);
	}

	//! move operator
	void operator=(D3D11RenderTarget&& other) {
		swap(*this, other);
	}



	//! adl swap
	friend void swap(D3D11RenderTarget& a, D3D11RenderTarget& b) {
		std::swap(a.m_graphics, b.m_graphics);
		std::swap(a.m_width, b.m_width);
		std::swap(a.m_height, b.m_height);
		std::swap(a.m_textureFormats, b.m_textureFormats);
		std::swap(a.m_targets, b.m_targets);
		std::swap(a.m_targetsRTV, b.m_targetsRTV);
		std::swap(a.m_targetsSRV, b.m_targetsSRV);
		std::swap(a.m_depthStencil, b.m_depthStencil);
		std::swap(a.m_depthStencilDSV, b.m_depthStencilDSV);
		std::swap(a.m_depthStencilSRV, b.m_depthStencilSRV);
		std::swap(a.m_captureTextures, b.m_captureTextures);
		std::swap(a.m_captureDepth, b.m_captureDepth);
		std::swap(a.m_bHasSRVs, b.m_bHasSRVs);
	}


    // create a new render target with given width and height. Also creates an equal-sized depth buffer.
	void init(GraphicsDevice &g, unsigned int width, unsigned int height, const std::vector<DXGI_FORMAT>& formats = std::vector < DXGI_FORMAT > {DXGI_FORMAT_R8G8B8A8_UNORM}, bool createSRVs = false) {
		m_graphics = &g.castD3D11();
		m_width = width;
		m_height = height;

		m_textureFormats = formats;
		m_bHasSRVs = createSRVs;

		createGPU();
	}

	void releaseGPU();
	void createGPU();

    // sets the render and depth buffers as the render target for the current device.
    // to return to the original graphics device render target, call bindRenderTarget() on the graphics device.
    void bind();

	//! restores the default render target
	void unbind() {
		m_graphics->bindRenderTarget();
	}

    // clears the render and depth buffers
    void clear(const vec4f& clearColor = vec4f(0.0f), float clearDepth = 1.0f);
    void clearColor(const vec4f& clearColor = vec4f(0.0f));
	void clearDepth(float clearDepth = 1.0f);
	  
	// get the i-th color buffer; could be ColorImageR8G8B8A8 or ColorImageR32G32B32A32
	template <class T>	void captureColorBuffer(BaseImage<T>& result, unsigned int which = 0);
	void captureDepthBuffer(DepthImage32& result);										//get the raw depth buffer
	void captureDepthBuffer(DepthImage32& result, const mat4f& perspectiveTransform);	//transforms the depth back to camera space
	void captureDepthBuffer(PointImage& result, const mat4f& perspectiveTransform);		//transforms it back to 3d camera coordinate

    unsigned int getWidth() const { return m_width; }
    unsigned int getHeight() const { return m_height; }

	unsigned int getNumTargets() const {
		return (unsigned int)m_textureFormats.size();
	}
	bool hasSRVs() const {
		return m_bHasSRVs;
	}

	ID3D11ShaderResourceView* getColorSRV(unsigned int which = 0) {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		return m_targetsSRV[which];
	}
	ID3D11ShaderResourceView* getDepthSRV() {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		return m_depthStencilSRV;
	}

	void bindColorSRVs(unsigned int startSlot = 0) {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		m_graphics->getContext().PSSetShaderResources(startSlot, getNumTargets(), m_targetsSRV);
	}

	void bindDepthSRV(unsigned int startSlot = 0) {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		m_graphics->getContext().PSSetShaderResources(startSlot, 1, &m_depthStencilSRV);
	}

	void unbindColorSRVs(unsigned int startSlot = 0) {
		ID3D11ShaderResourceView* const srvs[] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
		m_graphics->getContext().PSSetShaderResources(startSlot, getNumTargets(), srvs);
	}

	void unbindDepthSRV(unsigned int startSlot = 0) {
		ID3D11ShaderResourceView* const srvs[] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
		m_graphics->getContext().PSSetShaderResources(startSlot, 1, srvs);
	}
private:
	D3D11GraphicsDevice* m_graphics;
	unsigned int m_width;
	unsigned int m_height;

	std::vector<DXGI_FORMAT>	m_textureFormats;

    ID3D11Texture2D**			m_targets;
	ID3D11RenderTargetView**	m_targetsRTV;
	ID3D11ShaderResourceView**	m_targetsSRV;

	ID3D11Texture2D*			m_depthStencil;
	ID3D11DepthStencilView*		m_depthStencilDSV;
	ID3D11ShaderResourceView*	m_depthStencilSRV;

    ID3D11Texture2D** m_captureTextures;    
    ID3D11Texture2D* m_captureDepth;   
    
	bool m_bHasSRVs;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11RENDERTARGET_H_
