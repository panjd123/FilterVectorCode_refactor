#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <mutex>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include "config.h"


namespace ANNS {

    class NeighborList {
        public:
            static constexpr size_t INLINE_CAPACITY = 64;

            using value_type = IdxType;
            using iterator = IdxType*;
            using const_iterator = const IdxType*;

            NeighborList() = default;

            NeighborList(const NeighborList& other) {
                assign(other.begin(), other.end());
            }

            NeighborList(NeighborList&& other) noexcept {
                move_from(std::move(other));
            }

            NeighborList& operator=(const NeighborList& other) {
                if (this != &other)
                    assign(other.begin(), other.end());
                return *this;
            }

            NeighborList& operator=(NeighborList&& other) noexcept {
                if (this != &other) {
                    release_heap();
                    move_from(std::move(other));
                }
                return *this;
            }

            NeighborList& operator=(const std::vector<IdxType>& other) {
                assign(other.begin(), other.end());
                return *this;
            }

            ~NeighborList() {
                release_heap();
            }

            iterator begin() { return data_; }
            iterator end() { return data_ + size_; }
            const_iterator begin() const { return data_; }
            const_iterator end() const { return data_ + size_; }
            const_iterator cbegin() const { return data_; }
            const_iterator cend() const { return data_ + size_; }

            bool empty() const { return size_ == 0; }
            size_t size() const { return size_; }
            size_t capacity() const { return capacity_; }

            IdxType* data() { return data_; }
            const IdxType* data() const { return data_; }

            IdxType& operator[](size_t idx) { return data_[idx]; }
            const IdxType& operator[](size_t idx) const { return data_[idx]; }

            void clear() {
                size_ = 0;
            }

            void reserve(size_t new_capacity) {
                if (new_capacity <= capacity_)
                    return;
                grow_to(new_capacity);
            }

            void resize(size_t new_size) {
                reserve(new_size);
                if (new_size > size_)
                    std::fill(data_ + size_, data_ + new_size, IdxType{});
                size_ = new_size;
            }

            void push_back(IdxType value) {
                if (size_ == capacity_)
                    grow_to(next_capacity(size_ + 1));
                data_[size_++] = value;
            }

            template <class... Args>
            void emplace_back(Args&&... args) {
                push_back(IdxType(std::forward<Args>(args)...));
            }

            template <class InputIt>
            iterator insert(iterator pos, InputIt first, InputIt last) {
                const size_t offset = static_cast<size_t>(pos - begin());
                if (offset > size_)
                    throw std::out_of_range("NeighborList::insert position out of range");
                const size_t count = static_cast<size_t>(std::distance(first, last));
                if (count == 0)
                    return begin() + offset;
                reserve(size_ + count);
                pos = begin() + offset;
                std::move_backward(pos, end(), end() + count);
                std::copy(first, last, pos);
                size_ += count;
                return pos;
            }

            void assign(const_iterator first, const_iterator last) {
                const size_t count = static_cast<size_t>(last - first);
                reserve(count);
                if (count > 0)
                    std::memcpy(data_, first, count * sizeof(IdxType));
                size_ = count;
            }

            template <class InputIt>
            void assign(InputIt first, InputIt last) {
                const size_t count = static_cast<size_t>(std::distance(first, last));
                reserve(count);
                std::copy(first, last, data_);
                size_ = count;
            }

        private:
            IdxType inline_data_[INLINE_CAPACITY]{};
            IdxType* data_ = inline_data_;
            size_t size_ = 0;
            size_t capacity_ = INLINE_CAPACITY;

            bool using_inline() const {
                return data_ == inline_data_;
            }

            size_t next_capacity(size_t min_capacity) const {
                size_t next = capacity_ + capacity_ / 2 + 1;
                if (next < min_capacity)
                    next = min_capacity;
                return next;
            }

            void grow_to(size_t new_capacity) {
                IdxType* new_data = new IdxType[new_capacity];
                if (size_ > 0)
                    std::memcpy(new_data, data_, size_ * sizeof(IdxType));
                release_heap();
                data_ = new_data;
                capacity_ = new_capacity;
            }

            void release_heap() {
                if (!using_inline())
                    delete[] data_;
                data_ = inline_data_;
                capacity_ = INLINE_CAPACITY;
            }

            void move_from(NeighborList&& other) {
                if (other.using_inline()) {
                    if (other.size_ > 0)
                        std::memcpy(inline_data_, other.inline_data_, other.size_ * sizeof(IdxType));
                    data_ = inline_data_;
                    size_ = other.size_;
                    capacity_ = INLINE_CAPACITY;
                } else {
                    data_ = other.data_;
                    size_ = other.size_;
                    capacity_ = other.capacity_;
                    other.data_ = other.inline_data_;
                    other.size_ = 0;
                    other.capacity_ = INLINE_CAPACITY;
                }
            }
    };

    class Graph {
            
        public:
            NeighborList* neighbors;
            std::mutex* neighbor_locks;

            Graph() = default;

            Graph(IdxType num_points) {
                _num_points = num_points;
                neighbors = new NeighborList[num_points];
                neighbor_locks = new std::mutex[num_points];
            };

            Graph(std::shared_ptr<Graph> graph, IdxType start, IdxType end) {
                neighbors = graph->neighbors + start;
                neighbor_locks = graph->neighbor_locks + start;
                _num_points = end - start;
            };

            void save(std::string& filename) {
                std::ofstream out(filename);
                for (IdxType i = 0; i < _num_points; i++) {
                    out << i << " ";
                    for (auto& neighbor : neighbors[i])
                        out << neighbor << " ";
                    out << std::endl;
                }
                out.close();
            }

            void load(std::string& filename) {
                std::ifstream in(filename);
                std::string line;
                IdxType id, neighbor;
                while (std::getline(in, line)) {
                    std::istringstream iss(line);
                    iss >> id;
                    neighbors[id].clear();
                    while (iss >> neighbor) 
                        neighbors[id].push_back(neighbor);
                }
                in.close();
            }

            float get_index_size() {
                float index_size = 0;
                for (IdxType i = 0; i < _num_points; i++)
                    index_size += neighbors[i].size() * sizeof(IdxType);
                return index_size;
            }

            void clean() {
                delete[] neighbors;
                delete[] neighbor_locks;
            }

            ~Graph() = default;

        private:

            IdxType _num_points;
            
    };
}

#endif // GRAPH_H
