#pragma once

#ifndef CORE_UTIL_TIMER_H_
#define CORE_UTIL_TIMER_H_
#include <stdio.h>
#include <string>
#include <iostream>
#include <ctime>
#include <chrono>
namespace ml {

class Timer
{
public:
	Timer(std::string func_name = "") : beg_(clock_::now()), _func_name(func_name) {}
	void reset()
	{
		beg_ = clock_::now();
	}

	double elapsed()
	{
		return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
	}

	double getElapsedTime()
	{
		return elapsed();
	}

	void print(std::string message = "")
	{
#ifdef TIMER
		double t = elapsed();
		printf("[TIMER][%s][%s] elasped second: %f\n", _func_name.c_str(), message.c_str(), t);
		reset();
#endif
	}

private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1>> second_;
	std::chrono::time_point<clock_> beg_;
	std::string _func_name;
};

class ComponentTimer
{
public:
	ComponentTimer(const std::string &prompt)
	{
		m_prompt = prompt;
		m_terminated = false;
		std::cout << "start " << prompt << std::endl;
	}
	~ComponentTimer()
	{
		if(!m_terminated) end();
	}
	void end()
	{
		m_terminated = true;
		std::cout << "end " << m_prompt << ", " << std::to_string(m_clock.getElapsedTime()) << "s" << std::endl;
	}

private:
	std::string m_prompt;
	Timer m_clock;
	bool m_terminated;
};



class FrameTimer
{
public:
	FrameTimer()
	{
		m_secondsPerFrame = 1.0 / 60.0;
	}
	void frame()
	{
		double elapsed = m_clock.getElapsedTime();
		m_secondsPerFrame = elapsed * 0.2 + m_secondsPerFrame * 0.8;
	}
	float framesPerSecond()
	{
		return 1.0f / (float)m_secondsPerFrame;
	}
	float secondsPerFrame()
	{
		return (float)m_secondsPerFrame;
	}

private:
	Timer m_clock;
	double m_secondsPerFrame;
};

} // namespace ml


#endif // CORE_UTIL_TIMER_H_
