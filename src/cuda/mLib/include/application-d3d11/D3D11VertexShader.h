
#ifndef APPLICATION_D3D11_D3D11VERTEXSHADER_H_
#define APPLICATION_D3D11_D3D11VERTEXSHADER_H_

namespace ml {

	class D3D11VertexShader : public GraphicsAsset
	{
	public:
		D3D11VertexShader()
		{
			m_shader = nullptr;
			m_blob = nullptr;
			m_standardLayout = nullptr;
			m_graphics = nullptr;
		}
		~D3D11VertexShader()
		{
			SAFE_RELEASE(m_shader);
			SAFE_RELEASE(m_blob);
			SAFE_RELEASE(m_standardLayout);
		}
		void init(
			GraphicsDevice &g, 
			const std::string &filename, 
			const std::string& entryPoint = "vertexShaderMain", 
			const std::string& shaderModel = "vs_4_0",
			const std::vector<std::pair<std::string, std::string>>& shaderMacros = std::vector<std::pair<std::string, std::string>>());

		void releaseGPU();
		void createGPU();

		void bind() const;

	private:
		D3D11GraphicsDevice *m_graphics;
		ID3D11VertexShader *m_shader;
		ID3DBlob *m_blob;
		std::string m_filename;

		ID3D11InputLayout *m_standardLayout;
	};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11VERTEXSHADER_H_
