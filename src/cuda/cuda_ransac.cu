/*
Authors: Bowen Wen
Contact: wenbowenxjtu@gmail.com
Created in 2021

Copyright (c) Rutgers University, 2021 All rights reserved.

Bowen Wen and Kostas Bekris. "BundleTrack: 6D Pose Tracking for Novel Objects
 without Instance or Category-Level 3D Models."
 In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2021.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Bowen Wen, Kostas Bekris, Rutgers University,
      nor the names of its contributors may be used to
      endorse or promote products derived from this software without
      specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "cuda_ransac.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "core-util/timer.h"


#define gone					1065353216
#define gsine_pi_over_eight		1053028117
#define gcosine_pi_over_eight   1064076127
#define gone_half				0.5f
#define gsmall_number			1.e-12f
#define gtiny_number			1.e-20f
#define gfour_gamma_squared		5.8284273147583007813f



union un { float f; unsigned int ui; };


__device__ void svd(
	float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,
		float &u11, float &u12, float &u13, float &u21, float &u22, float &u23, float &u31, float &u32, float &u33,
	float &s11,
	float &s22,
	float &s33,
	float &v11, float &v12, float &v13, float &v21, float &v22, float &v23, float &v31, float &v32, float &v33
)
{
	un Sa11, Sa21, Sa31, Sa12, Sa22, Sa32, Sa13, Sa23, Sa33;
	un Su11, Su21, Su31, Su12, Su22, Su32, Su13, Su23, Su33;
	un Sv11, Sv21, Sv31, Sv12, Sv22, Sv32, Sv13, Sv23, Sv33;
	un Sc, Ss, Sch, Ssh;
	un Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
	un Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
	un Sqvs, Sqvvx, Sqvvy, Sqvvz;

	Sa11.f = a11; Sa12.f = a12; Sa13.f = a13;
	Sa21.f = a21; Sa22.f = a22; Sa23.f = a23;
	Sa31.f = a31; Sa32.f = a32; Sa33.f = a33;

	Ss11.f = Sa11.f*Sa11.f;
	Stmp1.f = Sa21.f*Sa21.f;
	Ss11.f = __fadd_rn(Stmp1.f, Ss11.f);
	Stmp1.f = Sa31.f*Sa31.f;
	Ss11.f = __fadd_rn(Stmp1.f, Ss11.f);

	Ss21.f = Sa12.f*Sa11.f;
	Stmp1.f = Sa22.f*Sa21.f;
	Ss21.f = __fadd_rn(Stmp1.f, Ss21.f);
	Stmp1.f = Sa32.f*Sa31.f;
	Ss21.f = __fadd_rn(Stmp1.f, Ss21.f);

	Ss31.f = Sa13.f*Sa11.f;
	Stmp1.f = Sa23.f*Sa21.f;
	Ss31.f = __fadd_rn(Stmp1.f, Ss31.f);
	Stmp1.f = Sa33.f*Sa31.f;
	Ss31.f = __fadd_rn(Stmp1.f, Ss31.f);

	Ss22.f = Sa12.f*Sa12.f;
	Stmp1.f = Sa22.f*Sa22.f;
	Ss22.f = __fadd_rn(Stmp1.f, Ss22.f);
	Stmp1.f = Sa32.f*Sa32.f;
	Ss22.f = __fadd_rn(Stmp1.f, Ss22.f);

	Ss32.f = Sa13.f*Sa12.f;
	Stmp1.f = Sa23.f*Sa22.f;
	Ss32.f = __fadd_rn(Stmp1.f, Ss32.f);
	Stmp1.f = Sa33.f*Sa32.f;
	Ss32.f = __fadd_rn(Stmp1.f, Ss32.f);

	Ss33.f = Sa13.f*Sa13.f;
	Stmp1.f = Sa23.f*Sa23.f;
	Ss33.f = __fadd_rn(Stmp1.f, Ss33.f);
	Stmp1.f = Sa33.f*Sa33.f;
	Ss33.f = __fadd_rn(Stmp1.f, Ss33.f);

	Sqvs.f = 1.f; Sqvvx.f = 0.f; Sqvvy.f = 0.f; Sqvvz.f = 0.f;

	for (int i = 0; i < 4; i++)
	{
		Ssh.f = Ss21.f * 0.5f;
		Stmp5.f = __fsub_rn(Ss11.f, Ss22.f);

		Stmp2.f = Ssh.f*Ssh.f;
		Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
		Ssh.ui = Stmp1.ui&Ssh.ui;
		Sch.ui = Stmp1.ui&Stmp5.ui;
		Stmp2.ui = ~Stmp1.ui&gone;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f*Ssh.f;
		Stmp2.f = Sch.f*Sch.f;
		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Stmp4.f = __frsqrt_rn(Stmp3.f);

		Ssh.f = Stmp4.f*Ssh.f;
		Sch.f = Stmp4.f*Sch.f;
		Stmp1.f = gfour_gamma_squared*Stmp1.f;
		Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

		Stmp2.ui = gsine_pi_over_eight&Stmp1.ui;
		Ssh.ui = ~Stmp1.ui&Ssh.ui;
		Ssh.ui = Ssh.ui | Stmp2.ui;
		Stmp2.ui = gcosine_pi_over_eight&Stmp1.ui;
		Sch.ui = ~Stmp1.ui&Sch.ui;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f * Ssh.f;
		Stmp2.f = Sch.f * Sch.f;
		Sc.f = __fsub_rn(Stmp2.f, Stmp1.f);
		Ss.f = Sch.f * Ssh.f;
		Ss.f = __fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f, Sch.f);
#endif

		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Ss33.f = Ss33.f * Stmp3.f;
		Ss31.f = Ss31.f * Stmp3.f;
		Ss32.f = Ss32.f * Stmp3.f;
		Ss33.f = Ss33.f * Stmp3.f;

		Stmp1.f = Ss.f * Ss31.f;
		Stmp2.f = Ss.f * Ss32.f;
		Ss31.f = Sc.f * Ss31.f;
		Ss32.f = Sc.f * Ss32.f;
		Ss31.f = __fadd_rn(Stmp2.f, Ss31.f);
		Ss32.f = __fsub_rn(Ss32.f, Stmp1.f);

		Stmp2.f = Ss.f*Ss.f;
		Stmp1.f = Ss22.f*Stmp2.f;
		Stmp3.f = Ss11.f*Stmp2.f;
		Stmp4.f = Sc.f*Sc.f;
		Ss11.f = Ss11.f*Stmp4.f;
		Ss22.f = Ss22.f*Stmp4.f;
		Ss11.f = __fadd_rn(Ss11.f, Stmp1.f);
		Ss22.f = __fadd_rn(Ss22.f, Stmp3.f);
		Stmp4.f = __fsub_rn(Stmp4.f, Stmp2.f);
		Stmp2.f = __fadd_rn(Ss21.f, Ss21.f);
		Ss21.f = Ss21.f*Stmp4.f;
		Stmp4.f = Sc.f*Ss.f;
		Stmp2.f = Stmp2.f*Stmp4.f;
		Stmp5.f = Stmp5.f*Stmp4.f;
		Ss11.f = __fadd_rn(Ss11.f, Stmp2.f);
		Ss21.f = __fsub_rn(Ss21.f, Stmp5.f);
		Ss22.f = __fsub_rn(Ss22.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("%.20g\n", Ss11.f);
		printf("%.20g %.20g\n", Ss21.f, Ss22.f);
		printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

		Stmp1.f = Ssh.f*Sqvvx.f;
		Stmp2.f = Ssh.f*Sqvvy.f;
		Stmp3.f = Ssh.f*Sqvvz.f;
		Ssh.f = Ssh.f*Sqvs.f;

		Sqvs.f = Sch.f*Sqvs.f;
		Sqvvx.f = Sch.f*Sqvvx.f;
		Sqvvy.f = Sch.f*Sqvvy.f;
		Sqvvz.f = Sch.f*Sqvvz.f;

		Sqvvz.f = __fadd_rn(Sqvvz.f, Ssh.f);
		Sqvs.f = __fsub_rn(Sqvs.f, Stmp3.f);
		Sqvvx.f = __fadd_rn(Sqvvx.f, Stmp2.f);
		Sqvvy.f = __fsub_rn(Sqvvy.f, Stmp1.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f, Sqvs.f);
#endif

		Ssh.f = Ss32.f * 0.5f;
		Stmp5.f = __fsub_rn(Ss22.f, Ss33.f);

		Stmp2.f = Ssh.f * Ssh.f;
		Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
		Ssh.ui = Stmp1.ui&Ssh.ui;
		Sch.ui = Stmp1.ui&Stmp5.ui;
		Stmp2.ui = ~Stmp1.ui&gone;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f * Ssh.f;
		Stmp2.f = Sch.f * Sch.f;
		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Stmp4.f = __frsqrt_rn(Stmp3.f);

		Ssh.f = Stmp4.f * Ssh.f;
		Sch.f = Stmp4.f * Sch.f;
		Stmp1.f = gfour_gamma_squared * Stmp1.f;
		Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

		Stmp2.ui = gsine_pi_over_eight&Stmp1.ui;
		Ssh.ui = ~Stmp1.ui&Ssh.ui;
		Ssh.ui = Ssh.ui | Stmp2.ui;
		Stmp2.ui = gcosine_pi_over_eight&Stmp1.ui;
		Sch.ui = ~Stmp1.ui&Sch.ui;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f * Ssh.f;
		Stmp2.f = Sch.f * Sch.f;
		Sc.f = __fsub_rn(Stmp2.f, Stmp1.f);
		Ss.f = Sch.f*Ssh.f;
		Ss.f = __fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f, Sch.f);
#endif

		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Ss11.f = Ss11.f * Stmp3.f;
		Ss21.f = Ss21.f * Stmp3.f;
		Ss31.f = Ss31.f * Stmp3.f;
		Ss11.f = Ss11.f * Stmp3.f;

		Stmp1.f = Ss.f*Ss21.f;
		Stmp2.f = Ss.f*Ss31.f;
		Ss21.f = Sc.f*Ss21.f;
		Ss31.f = Sc.f*Ss31.f;
		Ss21.f = __fadd_rn(Stmp2.f, Ss21.f);
		Ss31.f = __fsub_rn(Ss31.f, Stmp1.f);

		Stmp2.f = Ss.f*Ss.f;
		Stmp1.f = Ss33.f*Stmp2.f;
		Stmp3.f = Ss22.f*Stmp2.f;
		Stmp4.f = Sc.f * Sc.f;
		Ss22.f = Ss22.f * Stmp4.f;
		Ss33.f = Ss33.f * Stmp4.f;
		Ss22.f = __fadd_rn(Ss22.f, Stmp1.f);
		Ss33.f = __fadd_rn(Ss33.f, Stmp3.f);
		Stmp4.f = __fsub_rn(Stmp4.f, Stmp2.f);
		Stmp2.f = __fadd_rn(Ss32.f, Ss32.f);
		Ss32.f = Ss32.f*Stmp4.f;
		Stmp4.f = Sc.f*Ss.f;
		Stmp2.f = Stmp2.f*Stmp4.f;
		Stmp5.f = Stmp5.f*Stmp4.f;
		Ss22.f = __fadd_rn(Ss22.f, Stmp2.f);
		Ss32.f = __fsub_rn(Ss32.f, Stmp5.f);
		Ss33.f = __fsub_rn(Ss33.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("%.20g\n", Ss11.f);
		printf("%.20g %.20g\n", Ss21.f, Ss22.f);
		printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

		Stmp1.f = Ssh.f*Sqvvx.f;
		Stmp2.f = Ssh.f*Sqvvy.f;
		Stmp3.f = Ssh.f*Sqvvz.f;
		Ssh.f = Ssh.f*Sqvs.f;

		Sqvs.f = Sch.f*Sqvs.f;
		Sqvvx.f = Sch.f*Sqvvx.f;
		Sqvvy.f = Sch.f*Sqvvy.f;
		Sqvvz.f = Sch.f*Sqvvz.f;

		Sqvvx.f = __fadd_rn(Sqvvx.f, Ssh.f);
		Sqvs.f = __fsub_rn(Sqvs.f, Stmp1.f);
		Sqvvy.f = __fadd_rn(Sqvvy.f, Stmp3.f);
		Sqvvz.f = __fsub_rn(Sqvvz.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f, Sqvs.f);
#endif
#if 1

		Ssh.f = Ss31.f * 0.5f;
		Stmp5.f = __fsub_rn(Ss33.f, Ss11.f);

		Stmp2.f = Ssh.f*Ssh.f;
		Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
		Ssh.ui = Stmp1.ui&Ssh.ui;
		Sch.ui = Stmp1.ui&Stmp5.ui;
		Stmp2.ui = ~Stmp1.ui&gone;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f*Ssh.f;
		Stmp2.f = Sch.f*Sch.f;
		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Stmp4.f = __frsqrt_rn(Stmp3.f);

		Ssh.f = Stmp4.f*Ssh.f;
		Sch.f = Stmp4.f*Sch.f;
		Stmp1.f = gfour_gamma_squared*Stmp1.f;
		Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

		Stmp2.ui = gsine_pi_over_eight&Stmp1.ui;
		Ssh.ui = ~Stmp1.ui&Ssh.ui;
		Ssh.ui = Ssh.ui | Stmp2.ui;
		Stmp2.ui = gcosine_pi_over_eight&Stmp1.ui;
		Sch.ui = ~Stmp1.ui&Sch.ui;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f*Ssh.f;
		Stmp2.f = Sch.f*Sch.f;
		Sc.f = __fsub_rn(Stmp2.f, Stmp1.f);
		Ss.f = Sch.f*Ssh.f;
		Ss.f = __fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f, Sch.f);
#endif

		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Ss22.f = Ss22.f * Stmp3.f;
		Ss32.f = Ss32.f * Stmp3.f;
		Ss21.f = Ss21.f * Stmp3.f;
		Ss22.f = Ss22.f * Stmp3.f;

		Stmp1.f = Ss.f*Ss32.f;
		Stmp2.f = Ss.f*Ss21.f;
		Ss32.f = Sc.f*Ss32.f;
		Ss21.f = Sc.f*Ss21.f;
		Ss32.f = __fadd_rn(Stmp2.f, Ss32.f);
		Ss21.f = __fsub_rn(Ss21.f, Stmp1.f);

		Stmp2.f = Ss.f*Ss.f;
		Stmp1.f = Ss11.f*Stmp2.f;
		Stmp3.f = Ss33.f*Stmp2.f;
		Stmp4.f = Sc.f*Sc.f;
		Ss33.f = Ss33.f*Stmp4.f;
		Ss11.f = Ss11.f*Stmp4.f;
		Ss33.f = __fadd_rn(Ss33.f, Stmp1.f);
		Ss11.f = __fadd_rn(Ss11.f, Stmp3.f);
		Stmp4.f = __fsub_rn(Stmp4.f, Stmp2.f);
		Stmp2.f = __fadd_rn(Ss31.f, Ss31.f);
		Ss31.f = Ss31.f*Stmp4.f;
		Stmp4.f = Sc.f*Ss.f;
		Stmp2.f = Stmp2.f*Stmp4.f;
		Stmp5.f = Stmp5.f*Stmp4.f;
		Ss33.f = __fadd_rn(Ss33.f, Stmp2.f);
		Ss31.f = __fsub_rn(Ss31.f, Stmp5.f);
		Ss11.f = __fsub_rn(Ss11.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("%.20g\n", Ss11.f);
		printf("%.20g %.20g\n", Ss21.f, Ss22.f);
		printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

		Stmp1.f = Ssh.f*Sqvvx.f;
		Stmp2.f = Ssh.f*Sqvvy.f;
		Stmp3.f = Ssh.f*Sqvvz.f;
		Ssh.f = Ssh.f*Sqvs.f;

		Sqvs.f = Sch.f*Sqvs.f;
		Sqvvx.f = Sch.f*Sqvvx.f;
		Sqvvy.f = Sch.f*Sqvvy.f;
		Sqvvz.f = Sch.f*Sqvvz.f;

		Sqvvy.f = __fadd_rn(Sqvvy.f, Ssh.f);
		Sqvs.f = __fsub_rn(Sqvs.f, Stmp2.f);
		Sqvvz.f = __fadd_rn(Sqvvz.f, Stmp1.f);
		Sqvvx.f = __fsub_rn(Sqvvx.f, Stmp3.f);
#endif
	}

	Stmp2.f = Sqvs.f*Sqvs.f;
	Stmp1.f = Sqvvx.f*Sqvvx.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = Sqvvy.f*Sqvvy.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = Sqvvz.f*Sqvvz.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);

	Stmp1.f = __frsqrt_rn(Stmp2.f);
	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sqvs.f = Sqvs.f*Stmp1.f;
	Sqvvx.f = Sqvvx.f*Stmp1.f;
	Sqvvy.f = Sqvvy.f*Stmp1.f;
	Sqvvz.f = Sqvvz.f*Stmp1.f;

	Stmp1.f = Sqvvx.f*Sqvvx.f;
	Stmp2.f = Sqvvy.f*Sqvvy.f;
	Stmp3.f = Sqvvz.f*Sqvvz.f;
	Sv11.f = Sqvs.f*Sqvs.f;
	Sv22.f = __fsub_rn(Sv11.f, Stmp1.f);
	Sv33.f = __fsub_rn(Sv22.f, Stmp2.f);
	Sv33.f = __fadd_rn(Sv33.f, Stmp3.f);
	Sv22.f = __fadd_rn(Sv22.f, Stmp2.f);
	Sv22.f = __fsub_rn(Sv22.f, Stmp3.f);
	Sv11.f = __fadd_rn(Sv11.f, Stmp1.f);
	Sv11.f = __fsub_rn(Sv11.f, Stmp2.f);
	Sv11.f = __fsub_rn(Sv11.f, Stmp3.f);
	Stmp1.f = __fadd_rn(Sqvvx.f, Sqvvx.f);
	Stmp2.f = __fadd_rn(Sqvvy.f, Sqvvy.f);
	Stmp3.f = __fadd_rn(Sqvvz.f, Sqvvz.f);
	Sv32.f = Sqvs.f*Stmp1.f;
	Sv13.f = Sqvs.f*Stmp2.f;
	Sv21.f = Sqvs.f*Stmp3.f;
	Stmp1.f = Sqvvy.f*Stmp1.f;
	Stmp2.f = Sqvvz.f*Stmp2.f;
	Stmp3.f = Sqvvx.f*Stmp3.f;
	Sv12.f = __fsub_rn(Stmp1.f, Sv21.f);
	Sv23.f = __fsub_rn(Stmp2.f, Sv32.f);
	Sv31.f = __fsub_rn(Stmp3.f, Sv13.f);
	Sv21.f = __fadd_rn(Stmp1.f, Sv21.f);
	Sv32.f = __fadd_rn(Stmp2.f, Sv32.f);
	Sv13.f = __fadd_rn(Stmp3.f, Sv13.f);

	Stmp2.f = Sa12.f;
	Stmp3.f = Sa13.f;
	Sa12.f = Sv12.f*Sa11.f;
	Sa13.f = Sv13.f*Sa11.f;
	Sa11.f = Sv11.f*Sa11.f;
	Stmp1.f = Sv21.f*Stmp2.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp1.f);
	Stmp1.f = Sv31.f*Stmp3.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp1.f);
	Stmp1.f = Sv22.f*Stmp2.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp1.f);
	Stmp1.f = Sv32.f*Stmp3.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp1.f);
	Stmp1.f = Sv23.f*Stmp2.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp1.f);
	Stmp1.f = Sv33.f*Stmp3.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp1.f);

	Stmp2.f = Sa22.f;
	Stmp3.f = Sa23.f;
	Sa22.f = Sv12.f*Sa21.f;
	Sa23.f = Sv13.f*Sa21.f;
	Sa21.f = Sv11.f*Sa21.f;
	Stmp1.f = Sv21.f*Stmp2.f;
	Sa21.f = __fadd_rn(Sa21.f, Stmp1.f);
	Stmp1.f = Sv31.f*Stmp3.f;
	Sa21.f = __fadd_rn(Sa21.f, Stmp1.f);
	Stmp1.f = Sv22.f*Stmp2.f;
	Sa22.f = __fadd_rn(Sa22.f, Stmp1.f);
	Stmp1.f = Sv32.f*Stmp3.f;
	Sa22.f = __fadd_rn(Sa22.f, Stmp1.f);
	Stmp1.f = Sv23.f*Stmp2.f;
	Sa23.f = __fadd_rn(Sa23.f, Stmp1.f);
	Stmp1.f = Sv33.f*Stmp3.f;
	Sa23.f = __fadd_rn(Sa23.f, Stmp1.f);

	Stmp2.f = Sa32.f;
	Stmp3.f = Sa33.f;
	Sa32.f = Sv12.f*Sa31.f;
	Sa33.f = Sv13.f*Sa31.f;
	Sa31.f = Sv11.f*Sa31.f;
	Stmp1.f = Sv21.f*Stmp2.f;
	Sa31.f = __fadd_rn(Sa31.f, Stmp1.f);
	Stmp1.f = Sv31.f*Stmp3.f;
	Sa31.f = __fadd_rn(Sa31.f, Stmp1.f);
	Stmp1.f = Sv22.f*Stmp2.f;
	Sa32.f = __fadd_rn(Sa32.f, Stmp1.f);
	Stmp1.f = Sv32.f*Stmp3.f;
	Sa32.f = __fadd_rn(Sa32.f, Stmp1.f);
	Stmp1.f = Sv23.f*Stmp2.f;
	Sa33.f = __fadd_rn(Sa33.f, Stmp1.f);
	Stmp1.f = Sv33.f*Stmp3.f;
	Sa33.f = __fadd_rn(Sa33.f, Stmp1.f);

	Stmp1.f = Sa11.f*Sa11.f;
	Stmp4.f = Sa21.f*Sa21.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp4.f = Sa31.f*Sa31.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);

	Stmp2.f = Sa12.f*Sa12.f;
	Stmp4.f = Sa22.f*Sa22.f;
	Stmp2.f = __fadd_rn(Stmp2.f, Stmp4.f);
	Stmp4.f = Sa32.f*Sa32.f;
	Stmp2.f = __fadd_rn(Stmp2.f, Stmp4.f);

	Stmp3.f = Sa13.f*Sa13.f;
	Stmp4.f = Sa23.f*Sa23.f;
	Stmp3.f = __fadd_rn(Stmp3.f, Stmp4.f);
	Stmp4.f = Sa33.f*Sa33.f;
	Stmp3.f = __fadd_rn(Stmp3.f, Stmp4.f);

	Stmp4.ui = (Stmp1.f < Stmp2.f) ? 0xffffffff : 0;
	Stmp5.ui = Sa11.ui^Sa12.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa11.ui = Sa11.ui^Stmp5.ui;
	Sa12.ui = Sa12.ui^Stmp5.ui;

	Stmp5.ui = Sa21.ui^Sa22.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa21.ui = Sa21.ui^Stmp5.ui;
	Sa22.ui = Sa22.ui^Stmp5.ui;

	Stmp5.ui = Sa31.ui^Sa32.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa31.ui = Sa31.ui^Stmp5.ui;
	Sa32.ui = Sa32.ui^Stmp5.ui;

	Stmp5.ui = Sv11.ui^Sv12.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv11.ui = Sv11.ui^Stmp5.ui;
	Sv12.ui = Sv12.ui^Stmp5.ui;

	Stmp5.ui = Sv21.ui^Sv22.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv21.ui = Sv21.ui^Stmp5.ui;
	Sv22.ui = Sv22.ui^Stmp5.ui;

	Stmp5.ui = Sv31.ui^Sv32.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv31.ui = Sv31.ui^Stmp5.ui;
	Sv32.ui = Sv32.ui^Stmp5.ui;

	Stmp5.ui = Stmp1.ui^Stmp2.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp1.ui = Stmp1.ui^Stmp5.ui;
	Stmp2.ui = Stmp2.ui^Stmp5.ui;


	Stmp5.f = -2.f;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp4.f = 1.f;
	Stmp4.f = __fadd_rn(Stmp4.f, Stmp5.f);

	Sa12.f = Sa12.f*Stmp4.f;
	Sa22.f = Sa22.f*Stmp4.f;
	Sa32.f = Sa32.f*Stmp4.f;

	Sv12.f = Sv12.f*Stmp4.f;
	Sv22.f = Sv22.f*Stmp4.f;
	Sv32.f = Sv32.f*Stmp4.f;


	Stmp4.ui = (Stmp1.f < Stmp3.f) ? 0xffffffff : 0;
	Stmp5.ui = Sa11.ui^Sa13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa11.ui = Sa11.ui^Stmp5.ui;
	Sa13.ui = Sa13.ui^Stmp5.ui;

	Stmp5.ui = Sa21.ui^Sa23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa21.ui = Sa21.ui^Stmp5.ui;
	Sa23.ui = Sa23.ui^Stmp5.ui;

	Stmp5.ui = Sa31.ui^Sa33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa31.ui = Sa31.ui^Stmp5.ui;
	Sa33.ui = Sa33.ui^Stmp5.ui;

	Stmp5.ui = Sv11.ui^Sv13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv11.ui = Sv11.ui^Stmp5.ui;
	Sv13.ui = Sv13.ui^Stmp5.ui;

	Stmp5.ui = Sv21.ui^Sv23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv21.ui = Sv21.ui^Stmp5.ui;
	Sv23.ui = Sv23.ui^Stmp5.ui;

	Stmp5.ui = Sv31.ui^Sv33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv31.ui = Sv31.ui^Stmp5.ui;
	Sv33.ui = Sv33.ui^Stmp5.ui;

	Stmp5.ui = Stmp1.ui^Stmp3.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp1.ui = Stmp1.ui^Stmp5.ui;
	Stmp3.ui = Stmp3.ui^Stmp5.ui;


	Stmp5.f = -2.f;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp4.f = 1.f;
	Stmp4.f = __fadd_rn(Stmp4.f, Stmp5.f);

	Sa11.f = Sa11.f*Stmp4.f;
	Sa21.f = Sa21.f*Stmp4.f;
	Sa31.f = Sa31.f*Stmp4.f;

	Sv11.f = Sv11.f*Stmp4.f;
	Sv21.f = Sv21.f*Stmp4.f;
	Sv31.f = Sv31.f*Stmp4.f;


	Stmp4.ui = (Stmp2.f < Stmp3.f) ? 0xffffffff : 0;
	Stmp5.ui = Sa12.ui^Sa13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa12.ui = Sa12.ui^Stmp5.ui;
	Sa13.ui = Sa13.ui^Stmp5.ui;

	Stmp5.ui = Sa22.ui^Sa23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa22.ui = Sa22.ui^Stmp5.ui;
	Sa23.ui = Sa23.ui^Stmp5.ui;

	Stmp5.ui = Sa32.ui^Sa33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa32.ui = Sa32.ui^Stmp5.ui;
	Sa33.ui = Sa33.ui^Stmp5.ui;

	Stmp5.ui = Sv12.ui^Sv13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv12.ui = Sv12.ui^Stmp5.ui;
	Sv13.ui = Sv13.ui^Stmp5.ui;

	Stmp5.ui = Sv22.ui^Sv23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv22.ui = Sv22.ui^Stmp5.ui;
	Sv23.ui = Sv23.ui^Stmp5.ui;

	Stmp5.ui = Sv32.ui^Sv33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv32.ui = Sv32.ui^Stmp5.ui;
	Sv33.ui = Sv33.ui^Stmp5.ui;

	Stmp5.ui = Stmp2.ui^Stmp3.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp2.ui = Stmp2.ui^Stmp5.ui;
	Stmp3.ui = Stmp3.ui^Stmp5.ui;


	Stmp5.f = -2.f;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp4.f = 1.f;
	Stmp4.f = __fadd_rn(Stmp4.f, Stmp5.f);

	Sa13.f = Sa13.f*Stmp4.f;
	Sa23.f = Sa23.f*Stmp4.f;
	Sa33.f = Sa33.f*Stmp4.f;

	Sv13.f = Sv13.f*Stmp4.f;
	Sv23.f = Sv23.f*Stmp4.f;
	Sv33.f = Sv33.f*Stmp4.f;


	Su11.f = 1.f; Su12.f = 0.f; Su13.f = 0.f;
	Su21.f = 0.f; Su22.f = 1.f; Su23.f = 0.f;
	Su31.f = 0.f; Su32.f = 0.f; Su33.f = 1.f;

	Ssh.f = Sa21.f*Sa21.f;
	Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
	Ssh.ui = Ssh.ui&Sa21.ui;

	Stmp5.f = 0.f;
	Sch.f = __fsub_rn(Stmp5.f, Sa11.f);
	Sch.f = max(Sch.f, Sa11.f);
	Sch.f = max(Sch.f, gsmall_number);
	Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);
	Stmp1.f = Stmp1.f*Stmp2.f;

	Sch.f = __fadd_rn(Sch.f, Stmp1.f);

	Stmp1.ui = ~Stmp5.ui&Ssh.ui;
	Stmp2.ui = ~Stmp5.ui&Sch.ui;
	Sch.ui = Stmp5.ui&Sch.ui;
	Ssh.ui = Stmp5.ui&Ssh.ui;
	Sch.ui = Sch.ui | Stmp1.ui;
	Ssh.ui = Ssh.ui | Stmp2.ui;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sch.f = Sch.f*Stmp1.f;
	Ssh.f = Ssh.f*Stmp1.f;

	Sc.f = Sch.f*Sch.f;
	Ss.f = Ssh.f*Ssh.f;
	Sc.f = __fsub_rn(Sc.f, Ss.f);
	Ss.f = Ssh.f*Sch.f;
	Ss.f = __fadd_rn(Ss.f, Ss.f);


	Stmp1.f = Ss.f*Sa11.f;
	Stmp2.f = Ss.f*Sa21.f;
	Sa11.f = Sc.f*Sa11.f;
	Sa21.f = Sc.f*Sa21.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp2.f);
	Sa21.f = __fsub_rn(Sa21.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa12.f;
	Stmp2.f = Ss.f*Sa22.f;
	Sa12.f = Sc.f*Sa12.f;
	Sa22.f = Sc.f*Sa22.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp2.f);
	Sa22.f = __fsub_rn(Sa22.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa13.f;
	Stmp2.f = Ss.f*Sa23.f;
	Sa13.f = Sc.f*Sa13.f;
	Sa23.f = Sc.f*Sa23.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp2.f);
	Sa23.f = __fsub_rn(Sa23.f, Stmp1.f);


	Stmp1.f = Ss.f*Su11.f;
	Stmp2.f = Ss.f*Su12.f;
	Su11.f = Sc.f*Su11.f;
	Su12.f = Sc.f*Su12.f;
	Su11.f = __fadd_rn(Su11.f, Stmp2.f);
	Su12.f = __fsub_rn(Su12.f, Stmp1.f);

	Stmp1.f = Ss.f*Su21.f;
	Stmp2.f = Ss.f*Su22.f;
	Su21.f = Sc.f*Su21.f;
	Su22.f = Sc.f*Su22.f;
	Su21.f = __fadd_rn(Su21.f, Stmp2.f);
	Su22.f = __fsub_rn(Su22.f, Stmp1.f);

	Stmp1.f = Ss.f*Su31.f;
	Stmp2.f = Ss.f*Su32.f;
	Su31.f = Sc.f*Su31.f;
	Su32.f = Sc.f*Su32.f;
	Su31.f = __fadd_rn(Su31.f, Stmp2.f);
	Su32.f = __fsub_rn(Su32.f, Stmp1.f);


	Ssh.f = Sa31.f*Sa31.f;
	Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
	Ssh.ui = Ssh.ui&Sa31.ui;

	Stmp5.f = 0.f;
	Sch.f = __fsub_rn(Stmp5.f, Sa11.f);
	Sch.f = max(Sch.f, Sa11.f);
	Sch.f = max(Sch.f, gsmall_number);
	Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);
	Stmp1.f = Stmp1.f*Stmp2.f;

	Sch.f = __fadd_rn(Sch.f, Stmp1.f);

	Stmp1.ui = ~Stmp5.ui&Ssh.ui;
	Stmp2.ui = ~Stmp5.ui&Sch.ui;
	Sch.ui = Stmp5.ui&Sch.ui;
	Ssh.ui = Stmp5.ui&Ssh.ui;
	Sch.ui = Sch.ui | Stmp1.ui;
	Ssh.ui = Ssh.ui | Stmp2.ui;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sch.f = Sch.f*Stmp1.f;
	Ssh.f = Ssh.f*Stmp1.f;

	Sc.f = Sch.f*Sch.f;
	Ss.f = Ssh.f*Ssh.f;
	Sc.f = __fsub_rn(Sc.f, Ss.f);
	Ss.f = Ssh.f*Sch.f;
	Ss.f = __fadd_rn(Ss.f, Ss.f);


	Stmp1.f = Ss.f*Sa11.f;
	Stmp2.f = Ss.f*Sa31.f;
	Sa11.f = Sc.f*Sa11.f;
	Sa31.f = Sc.f*Sa31.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp2.f);
	Sa31.f = __fsub_rn(Sa31.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa12.f;
	Stmp2.f = Ss.f*Sa32.f;
	Sa12.f = Sc.f*Sa12.f;
	Sa32.f = Sc.f*Sa32.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp2.f);
	Sa32.f = __fsub_rn(Sa32.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa13.f;
	Stmp2.f = Ss.f*Sa33.f;
	Sa13.f = Sc.f*Sa13.f;
	Sa33.f = Sc.f*Sa33.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp2.f);
	Sa33.f = __fsub_rn(Sa33.f, Stmp1.f);


	Stmp1.f = Ss.f*Su11.f;
	Stmp2.f = Ss.f*Su13.f;
	Su11.f = Sc.f*Su11.f;
	Su13.f = Sc.f*Su13.f;
	Su11.f = __fadd_rn(Su11.f, Stmp2.f);
	Su13.f = __fsub_rn(Su13.f, Stmp1.f);

	Stmp1.f = Ss.f*Su21.f;
	Stmp2.f = Ss.f*Su23.f;
	Su21.f = Sc.f*Su21.f;
	Su23.f = Sc.f*Su23.f;
	Su21.f = __fadd_rn(Su21.f, Stmp2.f);
	Su23.f = __fsub_rn(Su23.f, Stmp1.f);

	Stmp1.f = Ss.f*Su31.f;
	Stmp2.f = Ss.f*Su33.f;
	Su31.f = Sc.f*Su31.f;
	Su33.f = Sc.f*Su33.f;
	Su31.f = __fadd_rn(Su31.f, Stmp2.f);
	Su33.f = __fsub_rn(Su33.f, Stmp1.f);


	Ssh.f = Sa32.f*Sa32.f;
	Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
	Ssh.ui = Ssh.ui&Sa32.ui;

	Stmp5.f = 0.f;
	Sch.f = __fsub_rn(Stmp5.f, Sa22.f);
	Sch.f = max(Sch.f, Sa22.f);
	Sch.f = max(Sch.f, gsmall_number);
	Stmp5.ui = (Sa22.f >= Stmp5.f) ? 0xffffffff : 0;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);
	Stmp1.f = Stmp1.f*Stmp2.f;

	Sch.f = __fadd_rn(Sch.f, Stmp1.f);

	Stmp1.ui = ~Stmp5.ui&Ssh.ui;
	Stmp2.ui = ~Stmp5.ui&Sch.ui;
	Sch.ui = Stmp5.ui&Sch.ui;
	Ssh.ui = Stmp5.ui&Ssh.ui;
	Sch.ui = Sch.ui | Stmp1.ui;
	Ssh.ui = Ssh.ui | Stmp2.ui;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sch.f = Sch.f*Stmp1.f;
	Ssh.f = Ssh.f*Stmp1.f;

	Sc.f = Sch.f*Sch.f;
	Ss.f = Ssh.f*Ssh.f;
	Sc.f = __fsub_rn(Sc.f, Ss.f);
	Ss.f = Ssh.f*Sch.f;
	Ss.f = __fadd_rn(Ss.f, Ss.f);


	Stmp1.f = Ss.f*Sa21.f;
	Stmp2.f = Ss.f*Sa31.f;
	Sa21.f = Sc.f*Sa21.f;
	Sa31.f = Sc.f*Sa31.f;
	Sa21.f = __fadd_rn(Sa21.f, Stmp2.f);
	Sa31.f = __fsub_rn(Sa31.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa22.f;
	Stmp2.f = Ss.f*Sa32.f;
	Sa22.f = Sc.f*Sa22.f;
	Sa32.f = Sc.f*Sa32.f;
	Sa22.f = __fadd_rn(Sa22.f, Stmp2.f);
	Sa32.f = __fsub_rn(Sa32.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa23.f;
	Stmp2.f = Ss.f*Sa33.f;
	Sa23.f = Sc.f*Sa23.f;
	Sa33.f = Sc.f*Sa33.f;
	Sa23.f = __fadd_rn(Sa23.f, Stmp2.f);
	Sa33.f = __fsub_rn(Sa33.f, Stmp1.f);


	Stmp1.f = Ss.f*Su12.f;
	Stmp2.f = Ss.f*Su13.f;
	Su12.f = Sc.f*Su12.f;
	Su13.f = Sc.f*Su13.f;
	Su12.f = __fadd_rn(Su12.f, Stmp2.f);
	Su13.f = __fsub_rn(Su13.f, Stmp1.f);

	Stmp1.f = Ss.f*Su22.f;
	Stmp2.f = Ss.f*Su23.f;
	Su22.f = Sc.f*Su22.f;
	Su23.f = Sc.f*Su23.f;
	Su22.f = __fadd_rn(Su22.f, Stmp2.f);
	Su23.f = __fsub_rn(Su23.f, Stmp1.f);

	Stmp1.f = Ss.f*Su32.f;
	Stmp2.f = Ss.f*Su33.f;
	Su32.f = Sc.f*Su32.f;
	Su33.f = Sc.f*Su33.f;
	Su32.f = __fadd_rn(Su32.f, Stmp2.f);
	Su33.f = __fsub_rn(Su33.f, Stmp1.f);

	v11 = Sv11.f; v12 = Sv12.f; v13 = Sv13.f;
	v21 = Sv21.f; v22 = Sv22.f; v23 = Sv23.f;
	v31 = Sv31.f; v32 = Sv32.f; v33 = Sv33.f;

	u11 = Su11.f; u12 = Su12.f; u13 = Su13.f;
	u21 = Su21.f; u22 = Su22.f; u23 = Su23.f;
	u31 = Su31.f; u32 = Su32.f; u33 = Su33.f;

	s11 = Sa11.f;
	s22 = Sa22.f;
	s33 = Sa33.f;
}


__device__ int evalPoseKernel(const float4 *ptsA, const float4 *ptsB, const int n_pts, const float4x4 &pose, const float dist_thres, int *inlier_ids)
{
	int inliers = 0;

	for (int i = 0; i < n_pts; i++)
	{
		float4 ptA_transformed = pose*ptsA[i];
		float dist = length(ptsB[i]-ptA_transformed);
		if (dist>dist_thres)
		{
			continue;
		}

		inlier_ids[inliers] = i;
		inliers++;
	}

	return inliers;
}

__device__ bool procrustesKernel(const float4 *src_samples, const float4 *dst_samples, const int n_pts, float4x4 &pose)
{
	pose.setIdentity();

	float3 src_mean = make_float3(0.0f, 0.0f, 0.0f);
	float3 dst_mean = make_float3(0.0f, 0.0f, 0.0f);

	for (int i=0;i<n_pts;i++)
	{
		src_mean.x += src_samples[i].x;
		src_mean.y += src_samples[i].y;
		src_mean.z += src_samples[i].z;

		dst_mean.x += dst_samples[i].x;
		dst_mean.y += dst_samples[i].y;
		dst_mean.z += dst_samples[i].z;
	}
	src_mean.x /= n_pts;
	src_mean.y /= n_pts;
	src_mean.z /= n_pts;

	dst_mean.x /= n_pts;
	dst_mean.y /= n_pts;


	dst_mean.z /= n_pts;

	float3x3 S;
	S.setZero();
	for (int i=0;i<n_pts;i++)
	{
		float sx = src_samples[i].x - src_mean.x;
		float sy = src_samples[i].y - src_mean.y;
		float sz = src_samples[i].z - src_mean.z;

		float dx = dst_samples[i].x - dst_mean.x;
		float dy = dst_samples[i].y - dst_mean.y;
		float dz = dst_samples[i].z - dst_mean.z;

		S(0,0) += sx * dx;
		S(0,1) += sx * dy;
		S(0,2) += sx * dz;
		S(1,0) += sy * dx;
		S(1,1) += sy * dy;
		S(1,2) += sy * dz;
		S(2,0) += sz * dx;
		S(2,1) += sz * dy;
		S(2,2) += sz * dz;
	}


	float3x3 U, V;
	float3 ss;
	svd(S(0,0),S(0,1),S(0,2),S(1,0),S(1,1),S(1,2),S(2,0),S(2,1),S(2,2),
		U(0,0),U(0,1),U(0,2),U(1,0),U(1,1),U(1,2),U(2,0),U(2,1),U(2,2),
		ss.x, ss.y, ss.z,
		V(0,0),V(0,1),V(0,2),V(1,0),V(1,1),V(1,2),V(2,0),V(2,1),V(2,2)
		);

	float3x3 R = V * U.getTranspose();
	float3x3 identity;
	identity.setIdentity();

	{
		float3x3 tmp = R.getTranspose()*R - identity;
		float diff = 0;
		for (int h=0;h<3;h++)
		{
			for (int w=0;w<3;w++)
			{
				diff += tmp(h,w)*tmp(h,w);
			}
		}
		diff = sqrt(diff);
		if (diff>=1e-3)
		{
			printf("R is not valid\n");
			return false;
		}

		if (R.det()<0)
		{
			for (int i=0;i<3;i++)
			{
				V(i,2) = -V(i,2);
			}
			R = V * U.getTranspose();
		}
	}

	for (int h=0;h<3;h++)
	{
		for (int w=0;w<3;w++)
		{
			pose(h,w) = R(h,w);
		}
	}

	float3 t = dst_mean - R*src_mean;
	pose(0,3) = t.x;
	pose(1,3) = t.y;
	pose(2,3) = t.z;

	return true;
}


__global__ void ransacMultiPairKernel(const float4 *ptsA, const float4 *ptsB, const int max_n_pts, const int n_pairs, const int *n_pts, const int4 *rand_list, const float dist_thres, const int n_trials, int *inlier_ids, int *n_inliers, float4x4 *poses)
{
	const int pair_id = blockIdx.y*blockDim.y + threadIdx.y;
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;

	if (pair_id>=n_pairs) return;

	if(trial_id >= n_trials) return;

	const int global_trial_id = pair_id*n_trials + trial_id;
	const int global_pts_id = pair_id*max_n_pts;

	int rand_idx[3];
	rand_idx[0] = rand_list[global_trial_id].x;
	rand_idx[1] = rand_list[global_trial_id].y;
	rand_idx[2] = rand_list[global_trial_id].z;

	if (rand_idx[0]==rand_idx[1] || rand_idx[1]==rand_idx[2] || rand_idx[0]==rand_idx[2]) return;
	if (rand_idx[0]<0 || rand_idx[1]<0 || rand_idx[2]<0) return;

	float4 src_samples[3];
	float4 dst_samples[3];

	for (int i = 0; i < 3; i++)
	{
		src_samples[i] = ptsA[rand_idx[i]];
		dst_samples[i] = ptsB[rand_idx[i]];
	}

	bool res = procrustesKernel(src_samples, dst_samples, 3, poses[global_trial_id]);

	if (!res)
	{
		return;
	}
	n_inliers[global_trial_id] = evalPoseKernel(ptsA+global_pts_id, ptsB+global_pts_id, n_pts[pair_id], poses[global_trial_id], dist_thres, inlier_ids+pair_id*max_n_pts*n_trials);
}



__global__ void ransacEstimateModelKernel(const float4* ptsA, const float4* ptsB, const int n_pts, const int n_trials, float4x4 *poses, int *isgood)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= n_trials)
	{
		return;
	}

	curandState state;
	curand_init(0, idx, 0, &state);

	int rand_idx[3];

	rand_idx[0] = round(curand_uniform(&state) * (n_pts-1));
	rand_idx[1] = round(curand_uniform(&state) * (n_pts-1));
	rand_idx[2] = round(curand_uniform(&state) * (n_pts-1));

	if (rand_idx[0]==rand_idx[1] || rand_idx[1]==rand_idx[2] || rand_idx[0]==rand_idx[2]) return;
	if (rand_idx[0]<0 || rand_idx[1]<0 || rand_idx[2]<0) return;

	float4 src_samples[3];
	float4 dst_samples[3];

	for (int i = 0; i < 3; i++)
	{
		src_samples[i] = ptsA[rand_idx[i]];
		dst_samples[i] = ptsB[rand_idx[i]];
	}

	bool res = procrustesKernel(src_samples, dst_samples, 3, poses[idx]);
	if (res)
	{
		isgood[idx] = 1;
	}

}

__global__ void ransacEvalModelKernel(const float4* ptsA, const float4* ptsB, const int n_pts, const float4x4 *poses, const int *isgood, const float dist_thres, const int n_trials, int *inlier_flags, int *n_inliers)
{
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;
	const int pt_id = blockIdx.y*blockDim.y + threadIdx.y;
	if (trial_id>=n_trials || pt_id>=n_pts) return;
	if (isgood[trial_id]==0) return;

	float4 ptA_transformed = poses[trial_id]*ptsA[pt_id];
	float dist = length(ptsB[pt_id]-ptA_transformed);
	if (dist>dist_thres)
	{
		return;
	}

	atomicAdd(n_inliers+trial_id, 1);
	inlier_flags[trial_id*n_pts+pt_id] = 1;

}

__global__ void findBestTrial(const int *n_inliers, const int n_trials, int *best_trial_num_inliers, int *best_trial_id)
{
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (trial_id>=n_trials) return;

	const int cur_n_inliers = n_inliers[trial_id];
	atomicMax(best_trial_num_inliers, cur_n_inliers);

	__syncthreads();

	if (cur_n_inliers==best_trial_num_inliers[0])
	{
		best_trial_id[0] = trial_id;
	}

}



void ransacGPU(const float4 *ptsA, const float4 *ptsB, const int n_pts, const std::vector<std::vector<int>> &sample_indices, const float dist_thres, Eigen::Matrix4f &best_pose, std::vector<int> &inlier_ids)
{

}



void ransacMultiPairGPU(const std::vector<float4*> &ptsA, const std::vector<float4*> &ptsB, const std::vector<int> &n_pts, const int n_trials, const float dist_thres, std::vector<std::vector<int>> &inlier_ids)
{
	const int n_frame_pairs = ptsA.size();
	inlier_ids.resize(n_frame_pairs);

	std::vector<int*> n_inliers_gpu(n_frame_pairs);
	std::vector<float4x4*> poses_gpu(n_frame_pairs);
	std::vector<int*> isgood_gpu(n_frame_pairs);
	std::vector<int*> inlier_flags_gpu(n_frame_pairs);
	std::vector<int*> best_trial_num_inliers_gpu(n_frame_pairs);
	std::vector<int*> best_trial_id_gpu(n_frame_pairs);


	for (int i=0;i<n_frame_pairs;i++)
	{
		const int n_pts_cur_pair = n_pts[i];
		cudaMalloc(&n_inliers_gpu[i], sizeof(int)*n_trials);
		cudaMalloc(&poses_gpu[i], sizeof(float4x4)*n_trials);
		cudaMalloc(&isgood_gpu[i], sizeof(int)*n_trials);
		cudaMalloc(&inlier_flags_gpu[i], sizeof(int)*n_trials*n_pts_cur_pair);
		cudaMalloc(&best_trial_num_inliers_gpu[i], sizeof(int));
		cudaMalloc(&best_trial_id_gpu[i], sizeof(int));


		std::vector<float4x4> poses(n_trials);
		for (int j=0;j<poses.size();j++)
		{
			poses[j].setIdentity();
		}
		cutilSafeCall(cudaMemcpy(poses_gpu[i], poses.data(), sizeof(float4x4)*n_trials, cudaMemcpyHostToDevice));
		cudaMemset(n_inliers_gpu[i], 0, sizeof(int)*n_trials);
		cudaMemset(isgood_gpu[i], 0, sizeof(int)*n_trials);
		cudaMemset(inlier_flags_gpu[i], 0, sizeof(int)*n_trials*n_pts_cur_pair);
		cudaMemset(best_trial_num_inliers_gpu[i], 0, sizeof(int));
		cudaMemset(best_trial_id_gpu[i], 0, sizeof(int));

	}
	cutilSafeCall(cudaDeviceSynchronize());

	cudaStream_t streams[n_frame_pairs];
	for (int i=0;i<n_frame_pairs;i++)
	{
		cudaStreamCreate(&streams[i]);
		const int n_pts_cur_pair = n_pts[i];
		int threads = 512;
		int blocks = (n_trials+threads-1)/threads;
		ransacEstimateModelKernel<<<blocks, threads, 0, streams[i]>>>(ptsA[i], ptsB[i], n_pts_cur_pair, n_trials, poses_gpu[i], isgood_gpu[i]);

		int threadsx = 32;
		int blocksx = (n_trials+threadsx-1)/threadsx;
		int threadsy = 32;
		int blocksy = (n_pts_cur_pair+threadsy-1)/threadsy;
		ransacEvalModelKernel<<<dim3(blocksx,blocksy), dim3(threadsx,threadsy), 0, streams[i]>>>(ptsA[i], ptsB[i], n_pts_cur_pair, poses_gpu[i], isgood_gpu[i], dist_thres, n_trials, inlier_flags_gpu[i], n_inliers_gpu[i]);


		findBestTrial<<<(n_trials+512-1)/512, 512, 0, streams[i]>>>(n_inliers_gpu[i], n_trials, best_trial_num_inliers_gpu[i], best_trial_id_gpu[i]);
	}

	for (int i=0;i<n_frame_pairs;i++)
	{
		cudaStreamSynchronize(streams[i]);
	}


	for (int i=0;i<n_frame_pairs;i++)
	{
		const int n_pts_cur_pair = n_pts[i];
		int best_trial_id = -1;
		cudaMemcpy(&best_trial_id, best_trial_id_gpu[i], sizeof(int), cudaMemcpyDeviceToHost);

		std::vector<int> inlier_flags(n_pts_cur_pair*n_trials, 0);
		cudaMemcpy(inlier_flags.data(), inlier_flags_gpu[i]+best_trial_id*n_pts_cur_pair, sizeof(int)*n_pts_cur_pair, cudaMemcpyDeviceToHost);
		inlier_ids[i].clear();
		inlier_ids[i].reserve(n_pts_cur_pair);
		for (int ii=0;ii<inlier_flags.size();ii++)
		{
			if (inlier_flags[ii]==1)
			{
				inlier_ids[i].push_back(ii);
			}
		}
	}


	for (int i=0;i<n_frame_pairs;i++)
	{
		cutilSafeCall(cudaFree(n_inliers_gpu[i]));
		cutilSafeCall(cudaFree(poses_gpu[i]));
		cutilSafeCall(cudaFree(isgood_gpu[i]));
		cutilSafeCall(cudaFree(inlier_flags_gpu[i]));
		cutilSafeCall(cudaFree(best_trial_num_inliers_gpu[i]));
		cutilSafeCall(cudaFree(best_trial_id_gpu[i]));

	}

}
