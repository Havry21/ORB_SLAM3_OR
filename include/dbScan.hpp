#pragma once
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <queue>

namespace Clustering
{

    class DBSCAN
    {
    public:
        DBSCAN(double eps, size_t minPts, double precision = 1e-6)
            : eps_(eps), minPts_(minPts), precision_(precision) {}

        struct ClusterResult
        {
            std::vector<std::vector<geometry_msgs::msg::Point>> clusters;
            std::vector<geometry_msgs::msg::Point> noise;
        };

        ClusterResult cluster(const std::vector<geometry_msgs::msg::Point>& points)
        {
            ClusterResult result;
            std::vector<bool> visited(points.size(), false);
            std::vector<bool> clustered(points.size(), false);

            for (size_t i = 0; i < points.size(); ++i)
            {
                if (visited[i])
                    continue;

                visited[i] = true;
                auto neighbors = regionQuery(points, points[i]);

                if (neighbors.size() < minPts_)
                {
                    result.noise.push_back(points[i]);
                }
                else
                {
                    std::vector<geometry_msgs::msg::Point> cluster;
                    expandCluster(points, i, neighbors, cluster, visited, clustered);
                    result.clusters.push_back(std::move(cluster));
                }
            }

            return result;
        }

    private:
        double eps_;
        size_t minPts_;
        double precision_;

        std::vector<size_t> regionQuery(
            const std::vector<geometry_msgs::msg::Point>& points,
            const geometry_msgs::msg::Point& point) const
        {

            std::vector<size_t> neighbors;
            for (size_t i = 0; i < points.size(); ++i)
            {
                if (distanceBetween(points[i], point) <= eps_)
                {
                    neighbors.push_back(i);
                }
            }
            return neighbors;
        }

        void expandCluster(
            const std::vector<geometry_msgs::msg::Point>& points,
            size_t point_idx,
            const std::vector<size_t>& neighbors,
            std::vector<geometry_msgs::msg::Point>& cluster,
            std::vector<bool>& visited,
            std::vector<bool>& clustered)
        {

            cluster.push_back(points[point_idx]);
            clustered[point_idx] = true;

            std::queue<size_t> queue;
            for (auto idx : neighbors)
            {
                queue.push(idx);
            }

            while (! queue.empty())
            {
                auto current_idx = queue.front();
                queue.pop();

                if (! visited[current_idx])
                {
                    visited[current_idx] = true;
                    auto current_neighbors = regionQuery(points, points[current_idx]);

                    if (current_neighbors.size() >= minPts_)
                    {
                        for (auto neighbor_idx : current_neighbors)
                        {
                            if (! visited[neighbor_idx])
                            {
                                queue.push(neighbor_idx);
                            }
                        }
                    }
                }

                if (! clustered[current_idx])
                {
                    cluster.push_back(points[current_idx]);
                    clustered[current_idx] = true;
                }
            }
        }

        double distanceBetween(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) const
        {
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            double dz = p1.z - p2.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        bool pointsEqual(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) const
        {
            return std::abs(p1.x - p2.x) < precision_ &&
                   std::abs(p1.y - p2.y) < precision_ &&
                   std::abs(p1.z - p2.z) < precision_;
        }
    };

} // namespace Clustering