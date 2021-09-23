
#ifndef APPLICATION_D3D11_D3D11POINTCLOUD_H_
#define APPLICATION_D3D11_D3D11POINTCLOUD_H_


namespace ml {

	class D3D11PointCloud : public GraphicsAsset
	{
	public:
		D3D11PointCloud() {
			m_graphics = nullptr;
			m_vertexBuffer = nullptr;
		}

		template<class T>
		D3D11PointCloud(GraphicsDevice& g, const PointCloud<T>& pointCloud) {
			m_vertexBuffer = nullptr;
			init(g, pointCloud);
		}

		//! copy constructor
		D3D11PointCloud(const D3D11PointCloud& t) {
			m_vertexBuffer = nullptr;
			init(*t.m_graphics, t);
		}
		//! move constructor
		D3D11PointCloud(D3D11PointCloud&& t) {
			m_graphics = nullptr;
			m_vertexBuffer = nullptr;
			swap(*this, t);
		}

		~D3D11PointCloud() {
			releaseGPU();
		}

		//! assignment operator
		void operator=(const D3D11PointCloud& t) {
			init(*t.m_graphics, t);
		}

		//! move operator
		void operator=(D3D11PointCloud&& t) {
			swap(*this, t);
		}

		//! adl swap
		friend void swap(D3D11PointCloud& a, D3D11PointCloud& b) {
			std::swap(a.m_graphics, b.m_graphics);
			std::swap(a.m_vertexBuffer, b.m_vertexBuffer);
			std::swap(a.m_points, b.m_points);
		}


		void init(GraphicsDevice& g, const D3D11PointCloud& pointCloud) {
			m_graphics = &g.castD3D11();
			m_points = pointCloud.m_points;
			createGPU();
		}

		template<class T>
		void init(GraphicsDevice& g, const PointCloud<T>& pointCloud) {
			m_graphics = &g.castD3D11();
			m_points.clear();
			m_points.reserve(pointCloud.m_points.size());
			for (size_t i = 0; i < pointCloud.m_points.size(); i++) {
				m_points.push_back(pointCloud.m_points[i]);
				auto& v = m_points.back();
				if (pointCloud.hasNormals()) v.normal = pointCloud.m_normals[i];
				if (pointCloud.hasColors()) v.color = pointCloud.m_colors[i];
				if (pointCloud.hasTexCoords()) v.texCoord = pointCloud.m_texCoords[i];
			}
			createGPU();
		}

		void releaseGPU();
		void createGPU();

		void render() const;


		//! Updates colors of this D3d11PointCloud to vertexColors. Precondition: vertexColors has same length as vertices otherwise exception is thrown
		void updateColors(const std::vector<vec4f>& vertexColors);

		//! computes and returns the bounding box; no caching
		BoundingBox3f computeBoundingBox() const {
			BoundingBox3f bbox;
			for (const auto& v : m_points) {
				bbox.include(v.position);
			}
			return bbox;
		}

		const std::vector<TriMeshf::Vertex>& getPoints() const {
			return m_points;
		}

		void getPointCloud(PointCloudf& pointCloud) const {
			pointCloud.clear();
			for (const auto& v : m_points) {
				pointCloud.m_points.push_back(v.position);
				pointCloud.m_colors.push_back(v.color);
				pointCloud.m_normals.push_back(v.normal);
			}
		}

	private:
		void initVB(GraphicsDevice& g);

		D3D11GraphicsDevice *m_graphics;
		ID3D11Buffer* m_vertexBuffer;
		std::vector<TriMeshf::Vertex> m_points;
	};

}  // namespace ml

#endif // APPLICATION_D3D11_D3D11POINTCLOUD_H_