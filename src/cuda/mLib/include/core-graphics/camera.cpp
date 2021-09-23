
namespace ml {


	template <class FloatType>
	Camera<FloatType>::Camera(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& worldUp, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar) {
		m_eye = eye;
		m_worldUp = worldUp.getNormalized();
		m_worldUp *= (FloatType)-1.0;	//compensate for projection matrix convention
		m_look = lookDir.getNormalized();
		m_right = (m_worldUp ^ m_look).getNormalized();
		
		MLIB_ASSERT(math::floatEqual(m_look, (m_right ^ m_worldUp)));

		m_up = m_worldUp;		

		m_fieldOfView = fieldOfView;
		m_aspect = aspect;
		m_zNear = zNear;
		m_zFar = zFar;

		m_projection = projMatrix(m_fieldOfView, m_aspect, m_zNear, m_zFar);

		update();
	}

	template <class FloatType>
	Camera<FloatType>::Camera(const Matrix4x4<FloatType>& cameraToWorld, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar) {
		m_eye		= vec3<FloatType>(cameraToWorld(0, 3), cameraToWorld(1, 3), cameraToWorld(2, 3));
		
		m_right		= vec3<FloatType>(cameraToWorld(0, 0), cameraToWorld(1, 0), cameraToWorld(2, 0));
		m_worldUp	= vec3<FloatType>(cameraToWorld(0, 1), cameraToWorld(1, 1), cameraToWorld(2, 1));		
		m_look		= vec3<FloatType>(cameraToWorld(0, 2), cameraToWorld(1, 2), cameraToWorld(2, 2));

		MLIB_ASSERT(math::floatEqual(m_look, (m_right ^ m_worldUp)));
	
		m_up = m_worldUp;

		m_fieldOfView = fieldOfView;
		m_aspect = aspect;
		m_zNear = zNear;
		m_zFar = zFar;

		m_projection = projMatrix(m_fieldOfView, m_aspect, m_zNear, m_zFar);

		update();
	}

	template <class FloatType>
	Camera<FloatType>::Camera(const std::string& str)	{
		std::istringstream s(str);
		auto read = [](std::istringstream &s, vec3<FloatType> &pt) {
			s >> pt.x >> pt.y >> pt.z;
		};
		read(s, m_eye);
		read(s, m_right);
		read(s, m_look);
		read(s, m_up);
		read(s, m_worldUp);
		s >> m_fieldOfView;
		s >> m_aspect;
		s >> m_zNear;
		s >> m_zFar;

		m_projection = projMatrix(m_fieldOfView, m_aspect, m_zNear, m_zFar);

		update();
	}

	template <class FloatType>
	std::string Camera<FloatType>::toString() const	{
		std::ostringstream s;
		auto write = [](std::ostringstream &s, const vec3<FloatType> &pt) {
			s << pt.x << ' ' << pt.y << ' ' << pt.z << ' ';
		};
		write(s, m_eye);
		write(s, m_right);
		write(s, m_look);
		write(s, m_up);
		write(s, m_worldUp);
		s << m_fieldOfView << ' ';
		s << m_aspect << ' ';
		s << m_zNear << ' ';
		s << m_zFar;
		return s.str();
	}

	template <class FloatType>
	void Camera<FloatType>::updateAspectRatio(FloatType newAspect) {
		m_aspect = newAspect;
		m_projection = projMatrix(m_fieldOfView, m_aspect, m_zNear, m_zFar);
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::updateWorldUp(const vec3<FloatType>& worldUp) {
		m_worldUp = -worldUp;	//the minus compensates for the projection matrix
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::updateFov(FloatType newFov) {
		m_fieldOfView = newFov;
		m_projection = projMatrix(m_fieldOfView, m_aspect, m_zNear, m_zFar);
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::update() {
		m_view = viewMatrix(m_eye, m_look, m_up, m_right);
		m_viewProjection = m_projection * m_view;
	}

	//! angle is specified in degrees
	template <class FloatType>
	void Camera<FloatType>::lookRight(FloatType theta) {
		applyTransform(Matrix3x3<FloatType>::rotation(m_worldUp, theta));
	}

	//! angle is specified in degrees
	template <class FloatType>
	void Camera<FloatType>::lookUp(FloatType theta) {
		applyTransform(Matrix3x3<FloatType>::rotation(m_right, -theta));
	}

	//! angle is specified in degrees
	template <class FloatType>
	void Camera<FloatType>::roll(FloatType theta) {
		applyTransform(Matrix3x3<FloatType>::rotation(m_look, theta));
	}

	template <class FloatType>
	void Camera<FloatType>::applyTransform(const Matrix3x3<FloatType>& transform) {
		m_up = transform * m_up;
		m_right = transform * m_right;
		m_look = transform * m_look;
		update();
	}
	template <class FloatType>
	void Camera<FloatType>::applyTransform(const Matrix4x4<FloatType>& transform) {
		const Matrix3x3<FloatType> rot = transform.getRotation();
		m_up = rot * m_up;
		m_right = rot * m_right;
		m_look = rot * m_look;
		m_eye += transform.getTranslation();
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::strafe(FloatType delta) {
		m_eye += m_right * delta;
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::jump(FloatType delta) {
		m_eye += m_up * delta;
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::move(FloatType delta) {
		m_eye += m_look * delta;
		update();
	}

    template <class FloatType>
    void Camera<FloatType>::translate(const vec3<FloatType> &v) {
        m_eye += v;
        update();
    }

	// resets given a new eye, lookDir, up, and right vector
	template <class FloatType>
	void Camera<FloatType>::reset(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& worldUp) {
		m_eye = eye;
		m_worldUp = worldUp.getNormalized();
		m_worldUp *= (FloatType)-1.0;	//compensate for projection matrix convention
		m_look = lookDir.getNormalized();
		m_right = (m_worldUp ^ m_look).getNormalized();

		MLIB_ASSERT(math::floatEqual(m_look, (m_right ^ m_worldUp)));

		m_up = m_worldUp;

		update();
	}

	template <class FloatType>
	Matrix4x4<FloatType> Camera<FloatType>::projMatrix(FloatType fieldOfView, FloatType aspectRatio, FloatType zNear, FloatType zFar) {
		FloatType width = 1.0f / tanf(math::degreesToRadians(fieldOfView) * 0.5f);
		FloatType height = aspectRatio / tanf(math::degreesToRadians(fieldOfView) * 0.5f);

		height *= -(FloatType)1.0;	//making it consistent with the kinect projection assumption that y pointing downwards

		return Matrix4x4<FloatType>(
			width, 0.0f, 0.0f, 0.0f,
			0.0f, height, 0.0f, 0.0f,
			0.0f, 0.0f, zFar / (zFar - zNear), zFar * zNear / (zNear - zFar),
			0.0f, 0.0f, 1.0f, 0.0f);
	}

	template <class FloatType>
	Matrix4x4<FloatType> Camera<FloatType>::viewMatrix(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& up, const vec3<FloatType>& right) {
		vec3<FloatType> l = lookDir.getNormalized();
		vec3<FloatType> u = up.getNormalized();
		vec3<FloatType> r = right.getNormalized();

		return Matrix4x4<FloatType>(r.x, r.y, r.z, -vec3<FloatType>::dot(r, eye),
			u.x, u.y, u.z, -vec3<FloatType>::dot(u, eye),
			l.x, l.y, l.z, -vec3<FloatType>::dot(l, eye),
			0.0f, 0.0f, 0.0f, 1.0f);
	}

	template <class FloatType>
	Ray<FloatType> Camera<FloatType>::getScreenRay(FloatType screenX, FloatType screenY) const	{
		return Ray<FloatType>(m_eye, getScreenRayDirection(screenX, screenY));
	}

	template <class FloatType>
	vec3<FloatType> Camera<FloatType>::getScreenRayDirection(FloatType screenX, FloatType screenY) const {
		
		vec3<FloatType> perspectivePoint(
			math::linearMap((FloatType)0.0, (FloatType)1.0, (FloatType)-1.0, (FloatType)1.0, screenX), 
			math::linearMap((FloatType)0.0, (FloatType)1.0, (FloatType)1.0, (FloatType)-1.0, screenY), 
			(FloatType)-0.5
			);

		return getViewProj().getInverse() * perspectivePoint - m_eye;
	}

}  // namespace ml
