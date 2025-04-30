#pragma once
#include <array>
#include <opencv2/opencv.hpp>

using IdMatPair = std::pair<int, cv::Mat>;
template <size_t N>
class CircularBuffer {
private:
    std::array<IdMatPair, N> buffer;
    size_t head = 0;
    size_t tail = 0;
    bool full = false;

public:
    void push(int id, const cv::Mat& mat) {
        buffer[tail] = IdMatPair(id, mat.clone());
        if (full) {
            head = (head + 1) % N;
        }
        tail = (tail + 1) % N;
        full = (tail == head);
    }

    IdMatPair pop() {
        if (empty()) {
            throw std::runtime_error("Buffer is empty");
        }
        IdMatPair value = buffer[head];
        head = (head + 1) % N;
        full = false;
        return value;
    }

    cv::Mat find_by_id(int target_id) const {
        for (size_t i = 0; i < size(); ++i) {
            size_t idx = (head + i) % N;
            if (buffer[idx].first == target_id) {
                return buffer[idx].second.clone();
            }
        }
        return cv::Mat();
    }

    bool empty() const {
        return (!full && (head == tail));
    }

    bool is_full() const {
        return full;
    }

    size_t size() const {
        if (full) return N;
        return (tail >= head) ? (tail - head) : (N + tail - head);
    }
};