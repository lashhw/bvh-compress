#include <iostream>
#include <cassert>
#include <vector>
#include <fstream>
#include <bvh/triangle.hpp>
#include <bvh/bvh.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#include "happly/happly.h"

constexpr size_t max_trig_in_leaf_size = 7;

typedef bvh::Bvh<float> bvh_t;
typedef bvh::Triangle<float> trig_t;
typedef bvh::Ray<float> ray_t;
typedef bvh::Vector3<float> vector_t;
typedef bvh::BoundingBox<float> bbox_t;
typedef bvh::SweepSahBuilder<bvh_t> builder_t;

typedef bvh_t::Node node_t;
typedef trig_t::Intersection intersection_t;

typedef bvh::SingleRayTraverser<bvh_t> traverser_t;
typedef bvh::ClosestPrimitiveIntersector<bvh_t, trig_t> primitive_intersector_t;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "usage: ./a.out MODEL_FILE RAY_FILE" << std::endl;
        exit(EXIT_FAILURE);
    }

    char* model_file = argv[1];
    char* ray_file = argv[2];
    std::cout << "MODEL_FILE = " << model_file << std::endl;
    std::cout << "RAY_FILE = " << ray_file << std::endl;

    happly::PLYData ply_data(model_file);
    std::vector<std::array<double, 3>> v_pos = ply_data.getVertexPositions();
    std::vector<std::vector<size_t>> f_idx = ply_data.getFaceIndices<size_t>();

    std::vector<trig_t> trigs;
    for (auto &face : f_idx) {
        trigs.emplace_back(vector_t((float)v_pos[face[0]][0], (float)v_pos[face[0]][1], (float)v_pos[face[0]][2]),
                           vector_t((float)v_pos[face[1]][0], (float)v_pos[face[1]][1], (float)v_pos[face[1]][2]),
                           vector_t((float)v_pos[face[2]][0], (float)v_pos[face[2]][1], (float)v_pos[face[2]][2]));
    }

    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(trigs.data(), trigs.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), trigs.size());
    std::cout << "global_bbox = ("
              << global_bbox.min[0] << ", " << global_bbox.min[1] << ", " << global_bbox.min[2] << "), ("
              << global_bbox.max[0] << ", " << global_bbox.max[1] << ", " << global_bbox.max[2] << ")" << std::endl;

    std::cout << "building..." << std::endl;

    bvh_t bvh_quant;
    builder_t builder_quant(bvh_quant);
    builder_quant.max_leaf_size = max_trig_in_leaf_size;
    builder_quant.build(global_bbox, bboxes.get(), centers.get(), trigs.size());

    std::queue<size_t> queue;
    queue.emplace(0);
    while (!queue.empty()) {
        size_t ref_idx = queue.front();
        node_t& curr_node = bvh_quant.nodes[ref_idx];
        queue.pop();

        size_t left_idx = curr_node.first_child_or_primitive;
        node_t& left_node = bvh_quant.nodes[left_idx];
        size_t right_idx = left_idx + 1;
        node_t& right_node = bvh_quant.nodes[right_idx];

        bbox_t curr_bbox = curr_node.bounding_box_proxy().to_bounding_box();
        float extent[3] = {
            (curr_node.bounds[1] - curr_bbox.min[0]),
            (curr_node.bounds[3] - curr_bbox.min[1]),
            (curr_node.bounds[5] - curr_bbox.min[2])
        };

        for (auto &x : extent)
            x = x == 0.0f ? 1.0f : x;

        float exp[3] = {
            std::powf(2, std::ceil(std::log2(extent[0] / 255.0f))),
            std::powf(2, std::ceil(std::log2(extent[1] / 255.0f))),
            std::powf(2, std::ceil(std::log2(extent[2] / 255.0f)))
        };

        if (!curr_node.is_leaf()) {
            for (auto &node : std::array<node_t*, 2>{&left_node, &right_node}) {
                for (int i = 0; i < 3; i++) {
                    float bound_quant_min_fp32 = std::floor((node->bounds[i * 2] - curr_bbox.min[i]) / exp[i]);
                    float bound_quant_max_fp32 = std::ceil((node->bounds[i * 2 + 1] - curr_bbox.min[i]) / exp[i]);
                    assert(0.0f <= bound_quant_min_fp32 && bound_quant_min_fp32 <= 255.0f);
                    assert(0.0f <= bound_quant_max_fp32 && bound_quant_max_fp32 <= 255.0f);

                    float old_bound_min = node->bounds[i * 2];
                    float old_bound_max = node->bounds[i * 2 + 1];

                    node->bounds[i * 2] = curr_bbox.min[i] + bound_quant_min_fp32 * exp[i];
                    node->bounds[i * 2 + 1] = curr_bbox.min[i] + bound_quant_max_fp32 * exp[i];

                    assert(node->bounds[i * 2] <= old_bound_min);
                    assert(node->bounds[i * 2 + 1] >= old_bound_max);
                }
            }
            queue.emplace(left_idx);
            queue.emplace(right_idx);
        }
    }

    bvh_t bvh;
    builder_t builder(bvh);
    builder.max_leaf_size = max_trig_in_leaf_size;
    builder.build(global_bbox, bboxes.get(), centers.get(), trigs.size());

    std::cout << "traversing..." << std::endl;

    traverser_t traverser(bvh);
    traverser_t traverser_quant(bvh_quant);

    primitive_intersector_t primitive_intersector(bvh, trigs.data());
    primitive_intersector_t primitive_intersector_quant(bvh_quant, trigs.data());

    traverser_t::Statistics statistics;
    traverser_t::Statistics statistics_quant;

    intmax_t correct_rays = 0;
    intmax_t total_rays = 0;

    std::ifstream ray_fs(ray_file);
    assert(ray_fs.is_open());

    for (float r[7]; ray_fs.read((char*)r, 7 * sizeof(float)); total_rays++) {
        ray_t ray(
            vector_t(r[0], r[1], r[2]),
            vector_t(r[3], r[4], r[5]),
            0.f,
            r[6]
        );
        auto result = traverser.traverse(ray, primitive_intersector, statistics);
        auto result_quant = traverser_quant.traverse(ray, primitive_intersector, statistics_quant);

        if (result.has_value()) {
            if (result_quant.has_value() &&
                result_quant->intersection.t == result->intersection.t &&
                result_quant->intersection.u == result->intersection.u &&
                result_quant->intersection.v == result->intersection.v)
                correct_rays++;
        } else if (!result_quant.has_value()) {
            correct_rays++;
        }
    }

    std::cout << "(vanilla)" << std::endl;
    std::cout << "  traversal_steps: " << statistics.traversal_steps << std::endl;
    std::cout << "  both_intersected: " << statistics.both_intersected << std::endl;
    std::cout << "  intersections_a: " << statistics.intersections_a << std::endl;
    std::cout << "  intersections_b: " << statistics.intersections_b << std::endl;
    std::cout << "  finalize: " << statistics.finalize << std::endl;

    std::cout << "(compressed)" << std::endl;
    std::cout << "  traversal_steps: " << statistics_quant.traversal_steps << std::endl;
    std::cout << "  both_intersected: " << statistics_quant.both_intersected << std::endl;
    std::cout << "  intersections_a: " << statistics_quant.intersections_a << std::endl;
    std::cout << "  intersections_b: " << statistics_quant.intersections_b << std::endl;
    std::cout << "  finalize: " << statistics_quant.finalize << std::endl;

    std::cout << "total_rays: " << total_rays << std::endl;
    std::cout << "correct_rays: " << correct_rays << std::endl;
}