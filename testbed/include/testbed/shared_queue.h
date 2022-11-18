#pragma once
#ifndef TESTBED_SHARED_QUEUE_H_
#define TESTBED_SHARED_QUEUE_H_

#include <testbed/common.h>

#include <condition_variable>
#include <deque>
#include <mutex>

TESTBED_NAMESPACE_BEGIN

class ICallable {
public:
    virtual ~ICallable() {}
    virtual void operator()() = 0;
};

template <typename T>
class Callable: public ICallable {
public:
    Callable() = default;
    explicit Callable(const T& _callable): m_callable(_callable) {}
private:
    T m_callable;
};

template <typename T>
std::unique_ptr<ICallable> callable(T&& _callable) {
    return std::make_unique<Callable<T>>(std::forward<T>(_callable));
}

class SharedQueueEmptyException {};

template <typename T>
class SharedQueue {
public:
    bool empty() const {
        std::lock_guard<std::mutex> lock{mMutex};
        return mRawQueue.empty();
    }
    size_t size() const {
        std::lock_guard<std::mutex> lock{mMutex};
        return mRawQueue.size();
    }

    void push(T&& newElem) {
        std::lock_guard<std::mutex> lock{mMutex};
        mRawQueue.emplace_back(std::forward<T>(newElem));
    }

    T tryPop(bool back = false) {
        std::unique_lock<std::mutex> lock{mMutex};

        if (mRawQueue.empty()) {
            throw SharedQueueEmptyException{};
        }

        if (back) {
            T result = std::move(mRawQueue.back());
            mRawQueue.pop_back();
            return result;
        } else {
            T result = std::move(mRawQueue.front());
            mRawQueue.pop_front();
            return result;
        }
    }

private:
    std::deque<T> mRawQueue;
    mutable std::mutex mMutex;
    std::condition_variable mDataCondition;
};

TESTBED_NAMESPACE_END

#endif // TESTBED_SHARED_QUEUE_H_