
#ifndef CORE_GRAPHICS_CAMERA_TRACKBALL_H_
#define CORE_GRAPHICS_CAMERA_TRACKBALL_H_

namespace ml {

//
// MATT: I am commenting out this class because it is out of-date with Camera -- it is easy to fix, but I am currently trying to get things to run with clang.  Feel free to un-comment this at any time.
//

/*template <class FloatType>
class CameraTrackball : public Camera<FloatType> {
public:
	CameraTrackball() {}
	CameraTrackball(const vec3<FloatType>& eye, const vec3<FloatType>& worldUp, const vec3<FloatType>& right, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar) :
		Camera<FloatType>(eye, worldUp, right, fieldOfView, aspect, zNear, zFar)
	{
		m_modelTranslation = vec3<FloatType>::origin;
		m_modelRotation.setIdentity();

		update();
	}

	//! zoom
	void moveModel(FloatType delta) {
        MLIB_ERROR("getLook() not defined");
		//m_modelTranslation += getLook() * delta;
		update();
	}
	//! rotate
	void rotateModel(const vec2<FloatType>& delta) {
		m_modelRotation = mat4f::rotation(getUp(), delta.x) * mat4f::rotation(getRight(), delta.y) * m_modelRotation;
		update();
	}
	void rotateModelUp(FloatType delta) {
		m_modelRotation = mat4f::rotation(getUp(), delta) * m_modelRotation;
		update();
	}
	void rotateModelRight(FloatType delta) {
		m_modelRotation = mat4f::rotation(getRight(), delta) * m_modelRotation;
		update();
	}

	void setModelTranslation(const vec3<FloatType>& t) {
		m_modelTranslation = t;
		update();
	}
	void setModelRotation(const Matrix4x4<FloatType>& r) {
		m_modelRotation = r;
		update();
	}
	void setModel(const Matrix4x4<FloatType>& t) {
		m_modelRotation.setIdentity();
		m_modelRotation.setRotation(t.getRotation());
		m_modelTranslation = t.getTranslation();
		update();
	}

	const vec3<FloatType>& getModelTranslation() const { return m_modelTranslation; }
	const Matrix4x4<FloatType>& getModelRotation() const { return m_modelRotation; }

	const Matrix4x4<FloatType>& getWorldViewProj() const { return m_worldViewProj; }
	const Matrix4x4<FloatType>& getWorldView() const { return m_worldView; }
	const Matrix4x4<FloatType>& getWorld() const { return m_world; }
	
	void updateAspectRatio(FloatType newAspect) {
		Camera<FloatType>::updateAspectRatio(newAspect);
		update();
	}

private:

	void update() {
		m_world = mat4f::translation(m_modelTranslation) * m_modelRotation;
		m_worldView = getCamera() * m_world;
		m_worldViewProj = getPerspective() * m_worldView;
	}

	vec3<FloatType> m_modelTranslation;
	Matrix4x4<FloatType> m_modelRotation;

	Matrix4x4<FloatType> m_worldViewProj;
	Matrix4x4<FloatType> m_worldView;
	Matrix4x4<FloatType> m_world;
};

typedef CameraTrackball<float> CameraTrackballf;
typedef CameraTrackball<double> CameraTrackballd;*/

}  // namespace ml


#endif  // CORE_GRAPHICS_CAMERA_H_
