#include "triangleface.h"

TriangleFace::TriangleFace(glm::vec3 a_, glm::vec3 b_, glm::vec3 c_, glm::vec3 a_n_, glm::vec3 b_n_, glm::vec3 c_n_, glm::vec2 a_uv_, glm::vec2 b_uv_, glm::vec2 c_uv_, bool use_per_vertex_normals_) :
    a(a_), b(b_), c(c_), a_n(a_n_), b_n(b_n_), c_n(c_n_), a_uv(a_uv_), b_uv(b_uv_), c_uv(c_uv_), use_per_vertex_normals(use_per_vertex_normals_)
{
    local_bbox.reset(new BoundingBox(glm::min(a,glm::min(b,c)),glm::max(a,glm::max(b,c))));
}

bool TriangleFace::IntersectLocal(const Ray &r, Intersection &i)
{
   // REQUIREMENT: Add triangle intersection code here.
   // it currently ignores all triangles and just returns false.
   //
   // Note that you are only intersecting a single triangle, and the vertices
   // of the triangle are supplied to you by the trimesh class.
   //
   // use_per_vertex_normals tells you if the triangle has per-vertex normals.
   // If it does, you should compute and use the Phong-interpolated normal at the intersection point.
   // If it does not, you should use the normal of the triangle's supporting plane.
   //
   // If the ray r intersects the triangle abc:
   // 1. put the hit parameter in i.t
   // 2. put the normal in i.normal
   // 3. put the texture coordinates in i.uv
   // and return true;
   //

   // Calculate the plane normal ((B - A) x (C - A))
   glm::dvec3 AB = {b.x - a.x, b.y - a.y, b.z - a.z};
   glm::dvec3 AC = {c.x - a.x, c.y - a.y, c.z - a.z};
   glm::dvec3 plane_normal = glm::cross(AB, AC);
   plane_normal = glm::normalize(plane_normal);

   glm::dvec3 d = r.direction;
   if (abs(glm::dot(plane_normal, d)) < EDGE_EPSILON) { // ray is parallel, no intersection
       return false;
   }
   glm::dvec3 p = r.position;
   double k = glm::dot(plane_normal, a);

   // Get the t intersect value
   i.t = (k - glm::dot(plane_normal, p)) / glm::dot(plane_normal, d);

   // Intersection point on the plane
   glm::dvec3 Q = r.at(i.t);

   // Check if Q isn't within the triangle ABC
   if (glm::dot(glm::cross((b - a), (Q - a)), plane_normal) < EDGE_EPSILON ||
          glm::dot(glm::cross((c - b), (Q - b)), plane_normal) < EDGE_EPSILON ||
          glm::dot(glm::cross((a - c), (Q - c)), plane_normal) < EDGE_EPSILON) {
       return false;
   }

   // Get the barycentric coordinate constants
   double alpha = glm::dot(glm::cross(c - b,Q - b), plane_normal) / glm::dot(glm::cross(b - a,c - a), plane_normal);
   double beta = glm::dot(glm::cross(a - c,Q - c), plane_normal) / glm::dot(glm::cross(b - a,c - a), plane_normal);
   double gamma = glm::dot(glm::cross(b - a,Q - a), plane_normal) / glm::dot(glm::cross(b - a,c - a), plane_normal);

   // Calculate the normal at point Q
   if (use_per_vertex_normals) {
      i.normal = glm::normalize(((float)alpha * a_n) + ((float)beta * b_n) + ((float)gamma * c_n));
   } else {
      i.normal = glm::normalize((alpha * plane_normal) + (beta * plane_normal) + (gamma * plane_normal));
   }

   // Calculate the texture coordinates at point Q
   glm::vec2 weighted_a_uv = {a_uv.x * alpha, a_uv.y * alpha};
   glm::vec2 weighted_b_uv = {b_uv.x * beta, b_uv.y * beta};
   glm::vec2 weighted_c_uv = {c_uv.x * gamma, c_uv.y * gamma};
   i.uv = weighted_a_uv + weighted_b_uv + weighted_c_uv;

   return true;
}
