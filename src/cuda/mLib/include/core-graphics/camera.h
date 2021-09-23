
#ifndef CORE_GRAPHICS_CAMERA_H_
#define CORE_GRAPHICS_CAMERA_H_

namespace ml {

	template <class FloatType>
	class Camera : public BinaryDataSerialize < Camera<FloatType> > {
	public:
		Camera() {}
		Camera(const std::string& s);

		//! Standard constructor (eye point, look direction, up vector)
		Camera(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& worldUp, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar);

		//! Construct camera from extrinsic cameraToWorld matrix (columns are x, y, z vectors and origin of camera in that order).
		//! If flipRight is set, flip the x coordinate; does set worldUp to the current up vector (might need to call updateWorlUp)
		Camera(const Matrix4x4<FloatType>& cameraToWorld, const FloatType fieldOfView, const FloatType aspect, const FloatType zNear, const FloatType zFar);

		void updateAspectRatio(FloatType newAspect);
		void updateWorldUp(const vec3<FloatType>& worldUp);

		void updateFov(FloatType newFov);
		void lookRight(FloatType theta);
		void lookUp(FloatType theta);
		void roll(FloatType theta);

		void strafe(FloatType delta);
		void jump(FloatType delta);
		void move(FloatType delta);
		void translate(const vec3<FloatType> &v);

		//! constructs a screen ray; screen coordinates are in [0; 1]
		Ray<FloatType> getScreenRay(FloatType screenX, FloatType screenY) const;
		//! constructs a screen dir; screen coordinates are in [0; 1]
		vec3<FloatType> getScreenRayDirection(FloatType screenX, FloatType screenY) const;

		//! returns the world-to-camera matrix
		Matrix4x4<FloatType> getView() const {
			return m_view;
		}

		//! returns the projection matrix
		Matrix4x4<FloatType> getProj() const {
			return m_projection;
		}

		//! returns the camera-projection matrix (world -> camera -> proj space)
		Matrix4x4<FloatType> getViewProj() const {
			return m_viewProjection;
		}

		//! returns the eye point
		vec3<FloatType> getEye() const {
			return m_eye;
		}

		//! returns the look direction
		vec3<FloatType> getLook() const {
			return m_look;
		}
		
		//! returns the right direction
		vec3<FloatType> getRight() const {
			return m_right;
		}

		//! returns the (current) up direction
		vec3<FloatType> getUp() const {
			return -m_up;
		}

		//! returns the work up direction (which is not necessarily the current up)
		vec3<FloatType> getWorldUp() const {
			return -m_worldUp;
		}

		FloatType getFoV() const {
			return m_fieldOfView;
		}

		FloatType getAspect() const {
			return m_aspect;
		}

		float getNearPlane() const {
			return m_zNear;
		}

		float getFarPlane() const {
			return m_zFar;
		}

		//! returns an intrinsic vision matrix
		Matrix4x4<FloatType> getIntrinsic(unsigned int width, unsigned int height) const {
			return graphicsToVisionProj(getProj(), width, height); 
		}

		//! returns an extrinsic vision matrix (it's the inverse of the view matrix, projecting from the current frame back to world)
		Matrix4x4<FloatType> getExtrinsic() const {
			return getView().getInverse();
		}
		
		std::string toString() const;

		void applyTransform(const Matrix3x3<FloatType>& transform);
		void applyTransform(const Matrix4x4<FloatType>& transform);
		void reset(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& worldUp);


		//! extrinsic is camera-to-world
		static Camera<FloatType> visionToGraphics(const Matrix4x4<FloatType>& extrinsic, unsigned int width, unsigned int height, FloatType fx, FloatType fy, FloatType zNear, FloatType zFar) {
			//not entirely sure whether there is a '-1' for width/height somewhere
			FloatType fov = (FloatType)2.0 * atan((FloatType)width / ((FloatType)2 * fx));
			FloatType aspect = (FloatType)width / (FloatType)height;

			return Camera<FloatType>(extrinsic, math::radiansToDegrees(fov), aspect, zNear, zFar);
		}

		static Matrix4x4<FloatType> visionToGraphicsProj(unsigned int width, unsigned int height, FloatType fx, FloatType fy, FloatType zNear, FloatType zFar) {
			//not entirely sure whether there is a '-1' for width/height somewhere
			FloatType fov = (FloatType)2.0 * atan((FloatType)width / ((FloatType)2 * fx));
			FloatType aspect = (FloatType)width / (FloatType)height;
			return projMatrix(math::radiansToDegrees(fov), aspect, zNear, zFar);
		}

		//! note: this assumes the DX11/OGl structure of m (see NDC space)
		static Matrix4x4<FloatType> graphicsToVisionProj(const Matrix4x4<FloatType>& proj, unsigned int width, unsigned int height) {
			FloatType fov = (FloatType)2.0 * atan((FloatType)1 / proj(0, 0));
			FloatType aspect = (FloatType)width / height;
			FloatType t = tan((FloatType)0.5 * fov);
			FloatType focalLengthX = (FloatType)0.5 * (FloatType)width / t;
			FloatType focalLengthY = (FloatType)0.5 * (FloatType)height / t * aspect;

			//focalLengthY = -focalLengthY;

			return  Matrix4x4<FloatType>(
				focalLengthX, 0.0f, (FloatType)(width - 1) / 2.0f, 0.0f,
				0.0f, focalLengthY, (FloatType)(height - 1) / 2.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f);
		}

		//!  constructs a projection matrix (field of view is in degrees); matrix is right-handed: assumes x->right, y->down, z->forward
		static Matrix4x4<FloatType> projMatrix(FloatType fieldOfView, FloatType aspectRatio, FloatType zNear, FloatType zFar);

		//! constructs a view matrix (world -> camera)
		static Matrix4x4<FloatType> viewMatrix(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& up, const vec3<FloatType>& right);

	private:
		void update();

		vec3<FloatType> m_eye, m_right, m_look, m_up;
		vec3<FloatType> m_worldUp;
		Matrix4x4<FloatType> m_view;
		Matrix4x4<FloatType> m_projection;
		Matrix4x4<FloatType> m_viewProjection;

		FloatType m_fieldOfView, m_aspect, m_zNear, m_zFar;
	};

	typedef Camera<float> Cameraf;
	typedef Camera<double> Camerad;

}  // namespace ml

#include "camera.cpp"

#endif  // CORE_GRAPHICS_CAMERA_H_
