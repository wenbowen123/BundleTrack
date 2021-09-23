
#ifndef APPLICATION_D3D11_D3D11TRIMESH_H_
#define APPLICATION_D3D11_D3D11TRIMESH_H_

namespace ml {

class D3D11TriMesh : public GraphicsAsset
{
public:
    static const UINT layoutElementCount = 4;	//accessed by D3D11VertexShader
    static const D3D11_INPUT_ELEMENT_DESC layout[layoutElementCount];

	D3D11TriMesh() {
        m_graphics = nullptr;
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
	}

	template<class T>
	D3D11TriMesh(GraphicsDevice& g, const MeshData<T>& meshData) {
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
		init(g, meshData);
	}

	template<class T>
	D3D11TriMesh(GraphicsDevice& g, const TriMesh<T>& triMesh) {
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
		init(g, triMesh);
	}

	//! copy constructor
	D3D11TriMesh(const D3D11TriMesh& t) {
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
		init(*t.m_graphics, t);
	}
	//! move constructor
	D3D11TriMesh(D3D11TriMesh&& t) {
		m_graphics = nullptr;
		m_vertexBuffer = nullptr;
		m_indexBuffer = nullptr;
		swap(*this, t);
	}

	~D3D11TriMesh() {
		releaseGPU();
	}
	
	//! assignment operator
	void operator=(const D3D11TriMesh& t) {
		init(*t.m_graphics, t);
	}

	//! move operator
	void operator=(D3D11TriMesh&& t) {
		swap(*this, t);
	}

	//! adl swap
	friend void swap(D3D11TriMesh& a, D3D11TriMesh& b) {
		std::swap(a.m_graphics, b.m_graphics);
		std::swap(a.m_vertexBuffer, b.m_vertexBuffer);
		std::swap(a.m_indexBuffer, b.m_indexBuffer);
		std::swap(a.m_triMesh, b.m_triMesh);
	}


    void init(GraphicsDevice& g, const D3D11TriMesh& mesh) {
		m_graphics = &g.castD3D11();
        m_triMesh = mesh.m_triMesh;
        createGPU();
    }

	template<class T>
	void init(GraphicsDevice& g, const TriMesh<T>& triMesh)	{
        m_graphics = &g.castD3D11();
        m_triMesh = triMesh;
		createGPU();
	}

	template<class T>
	void init(GraphicsDevice& g, const MeshData<T>& meshData) {
        init(g, TriMesh<T>(meshData));
	}

	void releaseGPU();
	void createGPU();

	void render() const;

	//! Updates colors of this D3D11TriMesh to vertexColors. Precondition: vertexColors has same length as vertices otherwise exception is thrown
	void updateColors(const std::vector<vec4f>& vertexColors);

	//! computes and returns the bounding box; no caching
	BoundingBox3f computeBoundingBox() const	{
        return m_triMesh.computeBoundingBox();
    }

	//! returns the trimesh on the CPU
    const TriMeshf& getTriMesh() const	{
        return m_triMesh;
	}

private:
    void initVB(GraphicsDevice &g);
	void initIB(GraphicsDevice &g);

	D3D11GraphicsDevice* m_graphics;
	ID3D11Buffer* m_vertexBuffer;
	ID3D11Buffer* m_indexBuffer;	
    TriMeshf m_triMesh;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11TRIMESH_H_