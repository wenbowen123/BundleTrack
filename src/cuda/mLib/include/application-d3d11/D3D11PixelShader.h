
#ifndef APPLICATION_D3D11_D3D11PIXELSHADER_H_
#define APPLICATION_D3D11_D3D11PIXELSHADER_H_

namespace ml {

	class D3D11PixelShader : public GraphicsAsset
	{
	public:
		D3D11PixelShader()
		{
			m_shader = nullptr;
			m_blob = nullptr;
			m_graphics = nullptr;
		}

		~D3D11PixelShader()
		{
			releaseGPU();
		}

		void init(
			GraphicsDevice &g, 
			const std::string &filename, 
			const std::string& entryPoint = "pixelShaderMain", 
			const std::string& shaderModel = "ps_4_0",
			const std::vector<std::pair<std::string, std::string>>& shaderMacros = std::vector<std::pair<std::string, std::string>>());

		void releaseGPU();
		void createGPU();

		void bind() const;

		UINT64 hash64();

	private:
		D3D11GraphicsDevice *m_graphics;
		ID3D11PixelShader *m_shader;
		ID3DBlob *m_blob;
		std::string m_filename;
	};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11PIXELSHADER_H_
