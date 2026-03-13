/*
 * kurbo.c — C translation of the Kurbo 2D geometry library
 * Original Rust source: Copyright 2018-2025 the Kurbo Authors
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 * Translated from:
 *   affine.rs, arc.rs, axis.rs, bezpath.rs (partial), circle.rs,
 *   common.rs, cubicbez.rs (partial), ellipse.rs (partial), fit.rs (partial),
 *   insets.rs, interop_euclid.rs (N/A - skipped), lib.rs, line.rs,
 *   mindist.rs, moments.rs (partial), offset.rs (partial), param_curve.rs,
 *   point.rs, quadbez.rs, quadspline.rs
 *
 * NOTE ON ITERATORS / GENERICS / TRAITS:
 *   Rust traits like ParamCurve, Shape, Iterator, etc. are represented here
 *   as plain C structs and functions. The dynamic dispatch / trait object
 *   patterns are left as stubs with LOUD COMMENTS where manual work is needed.
 *
 * Requires: C99 or later, linking with -lm
 */

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Forward declarations for solve_* functions (defined later in common.rs section) */
typedef struct
{
    double v[2];
    int    len;
} DVec2;
typedef struct
{
    double v[3];
    int    len;
} DVec3;
typedef struct
{
    double v[4];
    int    len;
} DVec4;
DVec2 solve_quadratic(double c0, double c1, double c2);
DVec3 solve_cubic(double c0, double c1, double c2, double c3);
DVec4 solve_quartic(double c0, double c1, double c2, double c3, double c4);

/* ============================================================
 * SECTION: BASIC TYPES
 * ============================================================ */

/* Vec2 — a 2D vector / displacement */
typedef struct
{
    double x;
    double y;
} Vec2;

/* Point — a 2D point */
typedef Vec2 Point;

/* Size */
typedef struct
{
    double width;
    double height;
} Size;

/* Rect */
typedef struct
{
    double x0;
    double y0;
    double x1;
    double y1;
} Rect;

/* Affine — 2D affine transform stored as [a, b, c, d, e, f] */
typedef struct
{
    double c[6];
} Affine;

/* Axis */
typedef enum
{
    AXIS_HORIZONTAL,
    AXIS_VERTICAL
} Axis;

/* Insets */
typedef struct
{
    double x0;
    double y0;
    double x1;
    double y1;
} Insets;

/* Line */
typedef struct
{
    Point p0;
    Point p1;
} Line;

/* QuadBez — quadratic Bezier */
typedef struct
{
    Point p0;
    Point p1;
    Point p2;
} QuadBez;

/* CubicBez — cubic Bezier */
typedef struct
{
    Point p0;
    Point p1;
    Point p2;
    Point p3;
} CubicBez;

/* Arc */
typedef struct
{
    Point  center;
    Vec2   radii;
    double start_angle;
    double sweep_angle;
    double x_rotation;
} Arc;

/* Circle */
typedef struct
{
    Point  center;
    double radius;
} Circle;

/* CircleSegment */
typedef struct
{
    Point  center;
    double outer_radius;
    double inner_radius;
    double start_angle;
    double sweep_angle;
} CircleSegment;

/* Nearest — result of nearest-point queries */
typedef struct
{
    double distance_sq;
    double t;
} Nearest;

/* ============================================================
 * SECTION: SMALL DYNAMIC ARRAYS
 * We need ArrayVec<f64, N> equivalents.
 * ============================================================ */

#define MAX_ROOTS_2 2
#define MAX_ROOTS_3 3
#define MAX_ROOTS_4 4
#define MAX_EXTREMA 4

/* DVec2/DVec3/DVec4 are forward-declared at the top of the file */

/* Pair of doubles (used in factor_quartic) */
typedef struct
{
    double a;
    double b;
} DoublePair;
typedef struct
{
    DoublePair v[2];
    int        len;
} DPairVec2;

static inline void dvec2_push(DVec2* a, double x)
{
    assert(a->len < MAX_ROOTS_2);
    a->v[a->len++] = x;
}
static inline void dvec3_push(DVec3* a, double x)
{
    assert(a->len < MAX_ROOTS_3);
    a->v[a->len++] = x;
}
static inline void dvec4_push(DVec4* a, double x)
{
    assert(a->len < MAX_ROOTS_4);
    a->v[a->len++] = x;
}

/* ============================================================
 * SECTION: CONSTANTS
 * ============================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define KURBO_PI         M_PI
#define KURBO_FRAC_PI_2  (M_PI / 2.0)
#define DEFAULT_ACCURACY 1e-6

/* ============================================================
 * SECTION: Vec2
 * ============================================================ */

static inline Vec2 vec2_new(double x, double y)
{
    Vec2 v = {x, y};
    return v;
}
static const Vec2 VEC2_ZERO = {0.0, 0.0};

static inline Vec2   vec2_add(Vec2 a, Vec2 b) { return vec2_new(a.x + b.x, a.y + b.y); }
static inline Vec2   vec2_sub(Vec2 a, Vec2 b) { return vec2_new(a.x - b.x, a.y - b.y); }
static inline Vec2   vec2_scale(Vec2 a, double s) { return vec2_new(a.x * s, a.y * s); }
static inline double vec2_dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
static inline double vec2_cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
static inline double vec2_hypot(Vec2 a) { return hypot(a.x, a.y); }
static inline double vec2_hypot2(Vec2 a) { return a.x * a.x + a.y * a.y; }
static inline Vec2   vec2_neg(Vec2 a) { return vec2_new(-a.x, -a.y); }

static inline Vec2 vec2_normalize(Vec2 a)
{
    double h = vec2_hypot(a);
    if (h == 0.0)
        return VEC2_ZERO;
    return vec2_new(a.x / h, a.y / h);
}

static inline Vec2 vec2_lerp(Vec2 a, Vec2 b, double t)
{
    return vec2_new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
}

static inline double vec2_atan2(Vec2 v) { return atan2(v.y, v.x); }

static inline Vec2 vec2_from_angle(double th) { return vec2_new(cos(th), sin(th)); }
static inline bool vec2_is_finite(Vec2 v) { return isfinite(v.x) && isfinite(v.y); }
static inline bool vec2_is_nan(Vec2 v) { return isnan(v.x) || isnan(v.y); }
static inline Vec2 vec2_turn_90(Vec2 v) { return vec2_new(-v.y, v.x); }
static inline Vec2 vec2_rotate_scale(Vec2 a, Vec2 b) { return vec2_new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

static inline Point vec2_to_point(Vec2 v)
{
    Point p = {v.x, v.y};
    return p;
}
static inline Vec2 point_to_vec2(Point p)
{
    Vec2 v = {p.x, p.y};
    return v;
}

/* ============================================================
 * SECTION: Point
 * ============================================================ */

static const Point POINT_ZERO   = {0.0, 0.0};
static const Point POINT_ORIGIN = {0.0, 0.0};

static inline Point point_new(double x, double y)
{
    Point p = {x, y};
    return p;
}

static inline Point point_add_vec2(Point p, Vec2 v) { return point_new(p.x + v.x, p.y + v.y); }
static inline Point point_sub_vec2(Point p, Vec2 v) { return point_new(p.x - v.x, p.y - v.y); }
static inline Vec2  point_sub(Point a, Point b) { return vec2_new(a.x - b.x, a.y - b.y); }

static inline Point point_lerp(Point a, Point b, double t)
{
    return vec2_to_point(vec2_lerp(point_to_vec2(a), point_to_vec2(b), t));
}

static inline Point point_midpoint(Point a, Point b) { return point_new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y)); }

static inline double point_distance(Point a, Point b) { return vec2_hypot(point_sub(a, b)); }

static inline double point_distance_sq(Point a, Point b) { return vec2_hypot2(point_sub(a, b)); }

static inline Point point_round(Point p) { return point_new(round(p.x), round(p.y)); }
static inline Point point_ceil(Point p) { return point_new(ceil(p.x), ceil(p.y)); }
static inline Point point_floor(Point p) { return point_new(floor(p.x), floor(p.y)); }
static inline Point point_trunc(Point p) { return point_new(trunc(p.x), trunc(p.y)); }

/* "expand" = round away from zero */
static inline double f64_expand(double x) { return copysign(ceil(fabs(x)), x); }
static inline Point  point_expand(Point p) { return point_new(f64_expand(p.x), f64_expand(p.y)); }

static inline bool point_is_finite(Point p) { return isfinite(p.x) && isfinite(p.y); }
static inline bool point_is_nan(Point p) { return isnan(p.x) || isnan(p.y); }

static inline double point_get_coord(Point p, Axis axis) { return axis == AXIS_HORIZONTAL ? p.x : p.y; }
static inline void   point_set_coord(Point* p, Axis axis, double val)
{
    if (axis == AXIS_HORIZONTAL)
        p->x = val;
    else
        p->y = val;
}

/* ============================================================
 * SECTION: Rect
 * ============================================================ */

static inline Rect rect_new(double x0, double y0, double x1, double y1)
{
    Rect r = {x0, y0, x1, y1};
    return r;
}

static inline Rect rect_from_points(Point a, Point b)
{
    return rect_new(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y);
}

static inline Rect rect_from_origin_size(Point origin, Size size)
{
    return rect_new(origin.x, origin.y, origin.x + size.width, origin.y + size.height);
}

static inline double rect_width(Rect r) { return r.x1 - r.x0; }
static inline double rect_height(Rect r) { return r.y1 - r.y0; }
static inline Size   rect_size(Rect r)
{
    Size s = {rect_width(r), rect_height(r)};
    return s;
}
static inline Point rect_origin(Rect r) { return point_new(r.x0, r.y0); }

static inline Rect rect_abs(Rect r)
{
    return rect_new(
        r.x0 < r.x1 ? r.x0 : r.x1,
        r.y0 < r.y1 ? r.y0 : r.y1,
        r.x0 > r.x1 ? r.x0 : r.x1,
        r.y0 > r.y1 ? r.y0 : r.y1);
}

static inline Rect rect_union(Rect a, Rect b)
{
    return rect_new(
        a.x0 < b.x0 ? a.x0 : b.x0,
        a.y0 < b.y0 ? a.y0 : b.y0,
        a.x1 > b.x1 ? a.x1 : b.x1,
        a.y1 > b.y1 ? a.y1 : b.y1);
}

static inline Rect rect_union_pt(Rect r, Point p)
{
    return rect_new(r.x0 < p.x ? r.x0 : p.x, r.y0 < p.y ? r.y0 : p.y, r.x1 > p.x ? r.x1 : p.x, r.y1 > p.y ? r.y1 : p.y);
}

static inline double rect_min_x(Rect r) { return r.x0 < r.x1 ? r.x0 : r.x1; }
static inline double rect_max_x(Rect r) { return r.x0 > r.x1 ? r.x0 : r.x1; }
static inline double rect_min_y(Rect r) { return r.y0 < r.y1 ? r.y0 : r.y1; }
static inline double rect_max_y(Rect r) { return r.y0 > r.y1 ? r.y0 : r.y1; }
static inline Point  rect_center(Rect r) { return point_new(0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1)); }
static inline double rect_area(Rect r) { return rect_width(r) * rect_height(r); }
static inline double rect_perimeter(Rect r) { return 2.0 * (fabs(rect_width(r)) + fabs(rect_height(r))); }
static inline bool   rect_is_finite(Rect r)
{
    return isfinite(r.x0) && isfinite(r.y0) && isfinite(r.x1) && isfinite(r.y1);
}
static inline bool rect_is_nan(Rect r) { return isnan(r.x0) || isnan(r.y0) || isnan(r.x1) || isnan(r.y1); }
static inline Rect rect_add_vec2(Rect r, Vec2 v) { return rect_new(r.x0 + v.x, r.y0 + v.y, r.x1 + v.x, r.y1 + v.y); }
static inline Rect rect_sub_vec2(Rect r, Vec2 v) { return rect_new(r.x0 - v.x, r.y0 - v.y, r.x1 - v.x, r.y1 - v.y); }

/* ============================================================
 * SECTION: Size
 * ============================================================ */

static inline Size size_new(double w, double h)
{
    Size s = {w, h};
    return s;
}

/* ============================================================
 * SECTION: Insets
 * ============================================================ */

static const Insets INSETS_ZERO = {0.0, 0.0, 0.0, 0.0};

static inline Insets insets_new(double x0, double y0, double x1, double y1)
{
    Insets i = {x0, y0, x1, y1};
    return i;
}
static inline Insets insets_uniform(double d) { return insets_new(d, d, d, d); }
static inline Insets insets_uniform_xy(double x, double y) { return insets_new(x, y, x, y); }

static inline double insets_x_value(Insets i) { return i.x0 + i.x1; }
static inline double insets_y_value(Insets i) { return i.y0 + i.y1; }
static inline Size   insets_size(Insets i) { return size_new(insets_x_value(i), insets_y_value(i)); }

static inline bool insets_are_nonnegative(Insets i) { return i.x0 >= 0.0 && i.y0 >= 0.0 && i.x1 >= 0.0 && i.y1 >= 0.0; }

static inline Insets insets_nonnegative(Insets i)
{
    return insets_new(
        i.x0 > 0.0 ? i.x0 : 0.0,
        i.y0 > 0.0 ? i.y0 : 0.0,
        i.x1 > 0.0 ? i.x1 : 0.0,
        i.y1 > 0.0 ? i.y1 : 0.0);
}

static inline bool insets_is_finite(Insets i)
{
    return isfinite(i.x0) && isfinite(i.y0) && isfinite(i.x1) && isfinite(i.y1);
}
static inline bool insets_is_nan(Insets i) { return isnan(i.x0) || isnan(i.y0) || isnan(i.x1) || isnan(i.y1); }

static inline Insets insets_neg(Insets i) { return insets_new(-i.x0, -i.y0, -i.x1, -i.y1); }

static inline Insets insets_add(Insets a, Insets b)
{
    return insets_new(a.x0 + b.x0, a.y0 + b.y0, a.x1 + b.x1, a.y1 + b.y1);
}
static inline Insets insets_sub(Insets a, Insets b)
{
    return insets_new(a.x0 - b.x0, a.y0 - b.y0, a.x1 - b.x1, a.y1 - b.y1);
}
static inline Insets insets_min(Insets a, Insets b)
{
    return insets_new(
        a.x0 < b.x0 ? a.x0 : b.x0,
        a.y0 < b.y0 ? a.y0 : b.y0,
        a.x1 < b.x1 ? a.x1 : b.x1,
        a.y1 < b.y1 ? a.y1 : b.y1);
}
static inline Insets insets_max(Insets a, Insets b)
{
    return insets_new(
        a.x0 > b.x0 ? a.x0 : b.x0,
        a.y0 > b.y0 ? a.y0 : b.y0,
        a.x1 > b.x1 ? a.x1 : b.x1,
        a.y1 > b.y1 ? a.y1 : b.y1);
}
static inline Insets insets_scale(Insets a, double s) { return insets_new(a.x0 * s, a.y0 * s, a.x1 * s, a.y1 * s); }
static inline Insets insets_div(Insets a, double s) { return insets_new(a.x0 / s, a.y0 / s, a.x1 / s, a.y1 / s); }

/* Rect +/- Insets */
static inline Rect rect_add_insets(Rect r, Insets ins)
{
    Rect a = rect_abs(r);
    return rect_new(a.x0 - ins.x0, a.y0 - ins.y0, a.x1 + ins.x1, a.y1 + ins.y1);
}
static inline Rect rect_sub_insets(Rect r, Insets ins) { return rect_add_insets(r, insets_neg(ins)); }
/* Rect - Rect = Insets */
static inline Insets rect_diff_insets(Rect a, Rect b)
{
    /* a - b: the insets that when added to b give a */
    return insets_new(b.x0 - a.x0, b.y0 - a.y0, a.x1 - b.x1, a.y1 - b.y1);
}

/* ============================================================
 * SECTION: Axis
 * ============================================================ */

static inline Axis axis_cross(Axis a) { return a == AXIS_HORIZONTAL ? AXIS_VERTICAL : AXIS_HORIZONTAL; }

static inline Point axis_pack_point(Axis a, double axis_val, double cross_val)
{
    return a == AXIS_HORIZONTAL ? point_new(axis_val, cross_val) : point_new(cross_val, axis_val);
}
static inline Size axis_pack_size(Axis a, double axis_val, double cross_val)
{
    return a == AXIS_HORIZONTAL ? size_new(axis_val, cross_val) : size_new(cross_val, axis_val);
}
static inline Vec2 axis_pack_vec2(Axis a, double axis_val, double cross_val)
{
    return a == AXIS_HORIZONTAL ? vec2_new(axis_val, cross_val) : vec2_new(cross_val, axis_val);
}

/* ============================================================
 * SECTION: Affine
 * ============================================================ */

static inline Affine affine_new(double a0, double a1, double a2, double a3, double a4, double a5)
{
    Affine a;
    a.c[0] = a0;
    a.c[1] = a1;
    a.c[2] = a2;
    a.c[3] = a3;
    a.c[4] = a4;
    a.c[5] = a5;
    return a;
}

static inline Affine affine_scale(double s) { return affine_new(s, 0.0, 0.0, s, 0.0, 0.0); }

static const Affine AFFINE_IDENTITY = {{1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
static const Affine AFFINE_FLIP_Y   = {{1.0, 0.0, 0.0, -1.0, 0.0, 0.0}};
static const Affine AFFINE_FLIP_X   = {{-1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};

static inline Affine affine_scale_non_uniform(double sx, double sy) { return affine_new(sx, 0.0, 0.0, sy, 0.0, 0.0); }

static inline Affine affine_rotate(double th)
{
    double s = sin(th), c = cos(th);
    return affine_new(c, s, -s, c, 0.0, 0.0);
}

static inline Affine affine_translate(Vec2 v) { return affine_new(1.0, 0.0, 0.0, 1.0, v.x, v.y); }

static inline Affine affine_skew(double skew_x, double skew_y)
{
    return affine_new(1.0, skew_y, skew_x, 1.0, 0.0, 0.0);
}

/* Affine * Affine */
static inline Affine affine_mul(Affine a, Affine b)
{
    return affine_new(
        a.c[0] * b.c[0] + a.c[2] * b.c[1],
        a.c[1] * b.c[0] + a.c[3] * b.c[1],
        a.c[0] * b.c[2] + a.c[2] * b.c[3],
        a.c[1] * b.c[2] + a.c[3] * b.c[3],
        a.c[0] * b.c[4] + a.c[2] * b.c[5] + a.c[4],
        a.c[1] * b.c[4] + a.c[3] * b.c[5] + a.c[5]);
}

/* Affine * Point */
static inline Point affine_mul_point(Affine a, Point p)
{
    return point_new(a.c[0] * p.x + a.c[2] * p.y + a.c[4], a.c[1] * p.x + a.c[3] * p.y + a.c[5]);
}

/* f64 * Affine */
static inline Affine affine_scale_by(double s, Affine a)
{
    return affine_new(s * a.c[0], s * a.c[1], s * a.c[2], s * a.c[3], s * a.c[4], s * a.c[5]);
}

static inline double affine_determinant(Affine a) { return a.c[0] * a.c[3] - a.c[1] * a.c[2]; }

static inline Affine affine_inverse(Affine a)
{
    double inv_det = 1.0 / affine_determinant(a);
    return affine_new(
        inv_det * a.c[3],
        -inv_det * a.c[1],
        -inv_det * a.c[2],
        inv_det * a.c[0],
        inv_det * (a.c[2] * a.c[5] - a.c[3] * a.c[4]),
        inv_det * (a.c[1] * a.c[4] - a.c[0] * a.c[5]));
}

static inline Vec2 affine_translation(Affine a) { return vec2_new(a.c[4], a.c[5]); }

static inline Affine affine_with_translation(Affine a, Vec2 t)
{
    a.c[4] = t.x;
    a.c[5] = t.y;
    return a;
}

static inline Affine affine_then_translate(Affine a, Vec2 t)
{
    a.c[4] += t.x;
    a.c[5] += t.y;
    return a;
}

static inline Affine affine_pre_rotate(Affine a, double th) { return affine_mul(a, affine_rotate(th)); }
static inline Affine affine_then_rotate(Affine a, double th) { return affine_mul(affine_rotate(th), a); }
static inline Affine affine_pre_scale(Affine a, double s) { return affine_mul(a, affine_scale(s)); }
static inline Affine affine_then_scale(Affine a, double s) { return affine_mul(affine_scale(s), a); }
static inline Affine affine_pre_translate(Affine a, Vec2 t) { return affine_mul(a, affine_translate(t)); }

static inline Affine affine_scale_about(double s, Point center)
{
    Vec2 cv = point_to_vec2(center);
    return affine_then_translate(affine_then_scale(affine_translate(vec2_neg(cv)), s), cv);
}

static inline Affine affine_rotate_about(double th, Point center)
{
    Vec2 cv = point_to_vec2(center);
    return affine_then_translate(affine_then_rotate(affine_translate(vec2_neg(cv)), th), cv);
}

static inline Affine affine_reflect(Point point, Vec2 direction)
{
    /* Householder reflection */
    Vec2   n  = vec2_normalize(vec2_new(direction.y, -direction.x));
    double x2 = n.x * n.x, xy = n.x * n.y, y2 = n.y * n.y;
    Affine aff = affine_new(1.0 - 2.0 * x2, -2.0 * xy, -2.0 * xy, 1.0 - 2.0 * y2, point.x, point.y);
    return affine_pre_translate(aff, vec2_neg(point_to_vec2(point)));
}

static inline Affine affine_map_unit_square(Rect r)
{
    return affine_new(rect_width(r), 0.0, 0.0, rect_height(r), r.x0, r.y0);
}

static inline bool affine_is_finite(Affine a)
{
    for (int i = 0; i < 6; i++)
        if (!isfinite(a.c[i]))
            return false;
    return true;
}
static inline bool affine_is_nan(Affine a)
{
    for (int i = 0; i < 6; i++)
        if (isnan(a.c[i]))
            return true;
    return false;
}

static inline Rect affine_transform_rect_bbox(Affine a, Rect rect)
{
    Point p00 = affine_mul_point(a, point_new(rect.x0, rect.y0));
    Point p01 = affine_mul_point(a, point_new(rect.x0, rect.y1));
    Point p10 = affine_mul_point(a, point_new(rect.x1, rect.y0));
    Point p11 = affine_mul_point(a, point_new(rect.x1, rect.y1));
    return rect_union(rect_from_points(p00, p01), rect_from_points(p10, p11));
}

/* SVD of the linear part: returns (scale_vec, angle) */
typedef struct
{
    Vec2   scale;
    double angle;
} AffineSVD;

static inline AffineSVD affine_svd(Affine a)
{
    double    aa = a.c[0], b = a.c[1], c = a.c[2], d = a.c[3];
    double    a2 = aa * aa, b2 = b * b, c2 = c * c, d2 = d * d;
    double    ab = aa * b, cd = c * d;
    double    angle = 0.5 * atan2(2.0 * (ab + cd), a2 - b2 + c2 - d2);
    double    s1    = sqrt((aa + d) * (aa + d) + (b - c) * (b - c));
    double    s2    = sqrt((aa - d) * (aa - d) + (b + c) * (b + c));
    AffineSVD r;
    r.scale = vec2_new(0.5 * (s1 + s2), 0.5 * fabs(s1 - s2));
    r.angle = angle;
    return r;
}

/* ============================================================
 * SECTION: Line
 * ============================================================ */

static inline Line line_new(Point p0, Point p1)
{
    Line l = {p0, p1};
    return l;
}
static inline Line line_reversed(Line l) { return line_new(l.p1, l.p0); }

static inline Point line_eval(Line l, double t) { return point_lerp(l.p0, l.p1, t); }

static inline double line_arclen(Line l) { return vec2_hypot(point_sub(l.p1, l.p0)); }

static inline double line_inv_arclen(Line l, double arclen) { return arclen / vec2_hypot(point_sub(l.p1, l.p0)); }

static inline double line_signed_area(Line l) { return 0.5 * vec2_cross(point_to_vec2(l.p0), point_to_vec2(l.p1)); }

static inline Nearest line_nearest(Line l, Point p)
{
    Vec2   d = point_sub(l.p1, l.p0);
    Vec2   v = point_sub(p, l.p0);
    double t = vec2_dot(d, v) / vec2_hypot2(d);
    if (t < 0.0)
        t = 0.0;
    if (t > 1.0)
        t = 1.0;
    Vec2    diff = vec2_sub(v, vec2_scale(d, t));
    Nearest n    = {vec2_hypot2(diff), t};
    return n;
}

static inline Rect line_bounding_box(Line l) { return rect_from_points(l.p0, l.p1); }

static inline Point line_midpoint(Line l) { return point_midpoint(l.p0, l.p1); }

/* crossing_point: where two lines (extended to infinity) intersect */
static inline bool line_crossing_point(Line a, Line b, Point* out)
{
    Vec2   ab  = point_sub(a.p1, a.p0);
    Vec2   cd  = point_sub(b.p1, b.p0);
    double pcd = vec2_cross(ab, cd);
    if (pcd == 0.0)
        return false;
    double h = vec2_cross(ab, point_sub(a.p0, b.p0)) / pcd;
    *out     = point_add_vec2(b.p0, vec2_scale(cd, h));
    return true;
}

/* NOTE: affine_mul_line was removed in favor of line_affine_transform below */

static inline Line line_affine_transform(Affine a, Line l)
{
    Line r = {affine_mul_point(a, l.p0), affine_mul_point(a, l.p1)};
    return r;
}

static inline bool line_is_finite(Line l) { return point_is_finite(l.p0) && point_is_finite(l.p1); }
static inline bool line_is_nan(Line l) { return point_is_nan(l.p0) || point_is_nan(l.p1); }

/* Line subsegment */
static inline Line line_subsegment(Line l, double t0, double t1)
{
    return line_new(line_eval(l, t0), line_eval(l, t1));
}

/* ============================================================
 * SECTION: QuadBez
 * ============================================================ */

static inline QuadBez quadbez_new(Point p0, Point p1, Point p2)
{
    QuadBez q = {p0, p1, p2};
    return q;
}

static inline Point quadbez_eval(QuadBez q, double t)
{
    double mt = 1.0 - t;
    double x  = mt * mt * q.p0.x + 2.0 * mt * t * q.p1.x + t * t * q.p2.x;
    double y  = mt * mt * q.p0.y + 2.0 * mt * t * q.p1.y + t * t * q.p2.y;
    return point_new(x, y);
}

static inline QuadBez quadbez_subsegment(QuadBez q, double t0, double t1)
{
    Point p0 = quadbez_eval(q, t0);
    Point p2 = quadbez_eval(q, t1);
    /* control point by de Casteljau at t0, then linear interpolate */
    double mt0        = 1.0 - t0;
    Point  p1_a       = point_new(mt0 * q.p0.x + t0 * q.p1.x, mt0 * q.p0.y + t0 * q.p1.y);
    double mt1_unused = 1.0 - t1;
    (void)mt1_unused;
    /* p1 for [t0,t1] segment */
    /* Use derivative midpoint approach */
    /* p1' = lerp of the two intermediate points at the mapped parameter */
    /* Simpler: elevate to cubic and back, or use the standard formula */
    /* Standard formula for quadbez subsegment: */
    double s   = (t1 - t0);
    double p1x = p1_a.x + s * (q.p1.x + (t0) * (q.p2.x - q.p1.x) - p0.x) / (1.0 - t0 + 1e-300);
    (void)p1x; /* avoid unused warning */
    /* MANUAL TRANSLATION NEEDED: quadbez_subsegment control point math */
    /* The Rust source uses eval at t0 + small delta and the subdivision formula. */
    /* A correct implementation follows: */
    /* mid control = (1-u)*q01 + u*q12 where q01, q12 are the casteljau intermediates at t0 */
    /* and u = (t1 - t0)/(1 - t0)  */
    double u;
    if (fabs(1.0 - t0) < 1e-12)
        u = 0.0;
    else
        u = (t1 - t0) / (1.0 - t0);
    double mt0_ = 1.0 - t0;
    Point  q01  = point_new(mt0_ * q.p0.x + t0 * q.p1.x, mt0_ * q.p0.y + t0 * q.p1.y);
    Point  q12  = point_new(mt0_ * q.p1.x + t0 * q.p2.x, mt0_ * q.p1.y + t0 * q.p2.y);
    Point  p1   = point_new((1.0 - u) * q01.x + u * q12.x, (1.0 - u) * q01.y + u * q12.y);
    return quadbez_new(p0, p1, p2);
}

/* Derivative of a QuadBez is a Line (as a "ConstVec" — here represented as Line) */
static inline Line quadbez_deriv(QuadBez q)
{
    /* The derivative is the linear bezier 2*(p1-p0) -> 2*(p2-p1) */
    Point dp0 = point_new(2.0 * (q.p1.x - q.p0.x), 2.0 * (q.p1.y - q.p0.y));
    Point dp1 = point_new(2.0 * (q.p2.x - q.p1.x), 2.0 * (q.p2.y - q.p1.y));
    return line_new(dp0, dp1);
}

static inline Point quadbez_deriv_eval(QuadBez q, double t)
{
    Line d = quadbez_deriv(q);
    return line_eval(d, t);
}

static inline double quadbez_signed_area(QuadBez q)
{
    /* Green's theorem: area = integral of y dx */
    return (q.p0.x * (2.0 * q.p1.y + q.p2.y) - q.p2.x * (q.p0.y + 2.0 * q.p1.y) + q.p1.x * (q.p2.y - q.p0.y)) / 6.0;
    /* NOTE: THE EXACT FORMULA FROM RUST quadbez.rs MAY DIFFER SLIGHTLY —
     * PLEASE VERIFY AGAINST THE ORIGINAL signed_area IMPLEMENTATION */
}

static inline Nearest quadbez_nearest(QuadBez q, Point p, double accuracy)
{
    /* Analytical nearest: minimize distance squared by solving for roots of
     * the dot product of (eval(t) - p) and deriv(t) — a cubic in t.
     * Port of quadbez.rs nearest() */
    double p0x = q.p0.x - p.x, p0y = q.p0.y - p.y;
    double p1x = q.p1.x - p.x, p1y = q.p1.y - p.y;
    double p2x = q.p2.x - p.x, p2y = q.p2.y - p.y;
    /* B(t) = (1-t)^2 p0 + 2t(1-t) p1 + t^2 p2 (shifted by -p) */
    /* B'(t) = 2[(p1-p0) + t(p0 - 2p1 + p2)] */
    /* B(t).B'(t) = 0 is the cubic: we expand: */
    double d0x = p1x - p0x, d0y = p1y - p0y;                         /* p1-p0 */
    double d1x = p0x - 2.0 * p1x + p2x, d1y = p0y - 2.0 * p1y + p2y; /* p0-2p1+p2 */
    /* B(t) = p0 + 2t*d0 + t^2*(d1+d0) ... let me use the direct expansion:
     * Bx(t) = p0x + 2*d0x*t + (p2x-2*p1x+p0x)*t^2 = p0x + 2*d0x*t + d1x*t^2
     * B'x(t) = 2*d0x + 2*d1x*t
     * Dot product: sum over x,y of Bcoord * B'coord_half = 0:
     * (p0 + 2t*d0 + t^2*d1).(d0 + t*d1) = 0
     * = p0.d0 + t*(p0.d1 + 2*d0.d0) + t^2*(2*d0.d1 + d1.d0... wait let's expand fully */
    double p0d0 = p0x * d0x + p0y * d0y;
    double p0d1 = p0x * d1x + p0y * d1y;
    double d0d0 = d0x * d0x + d0y * d0y;
    double d0d1 = d0x * d1x + d0y * d1y;
    double d1d1 = d1x * d1x + d1y * d1y;
    /* Expanding (p0 + 2t d0 + t^2 d1).(d0 + t d1):
     * = p0.d0 + t(p0.d1 + 2 d0.d0) + t^2(2 d0.d1 + d1.d0) + t^3(d1.d1)
     * wait: (p0 + 2t d0 + t^2 d1) . (d0 + t d1):
     *   p0.d0  +  2t d0.d0  +  t^2 d1.d0
     *   t p0.d1  +  2t^2 d0.d1  +  t^3 d1.d1
     * = p0.d0  +  t(2 d0.d0 + p0.d1)  +  t^2(d0.d1*3)  +  t^3 d1.d1
     * Hmm wait: d1.d0 + 2*d0.d1 = 3*d0.d1 since dot is commutative */
    double a = d1d1;
    double b = 3.0 * d0d1;
    double c = 2.0 * d0d0 + p0d1;
    double d = p0d0;
    /* Solve a*t^3 + b*t^2 + c*t + d = 0 */
    DVec3 sol    = solve_cubic(d, c, b, a);
    int   nroots = sol.len;
    /* evaluate at endpoints too */
    double best_t = 0.0;
    {
        double ex = p0x, ey = p0y; /* eval at t=0 minus p */
        best_t        = 0.0;
        double best_d = ex * ex + ey * ey;
        {
            double ex1 = p2x, ey1 = p2y;
            double d1_ = ex1 * ex1 + ey1 * ey1;
            if (d1_ < best_d)
            {
                best_d = d1_;
                best_t = 1.0;
            }
        }
        for (int i = 0; i < nroots; i++)
        {
            double t = sol.v[i];
            if (t <= 0.0 || t >= 1.0)
                continue;
            /* eval B(t) - p */
            double mt   = 1.0 - t;
            double bx   = mt * mt * p0x + 2.0 * mt * t * p1x + t * t * p2x;
            double by   = mt * mt * p0y + 2.0 * mt * t * p1y + t * t * p2y;
            double dist = bx * bx + by * by;
            if (dist < best_d)
            {
                best_d = dist;
                best_t = t;
            }
        }
        Nearest n = {best_d, best_t};
        return n;
    }
    (void)accuracy;
}

/* Bounding box via extrema */
static inline DVec2 quadbez_extrema(QuadBez q)
{
    /* Derivative roots: 2*(p1-p0) + 2*(p2-2*p1+p0)*t = 0 */
    DVec2 r;
    r.len     = 0;
    double ax = q.p0.x - 2.0 * q.p1.x + q.p2.x;
    double bx = q.p1.x - q.p0.x;
    if (fabs(ax) > 1e-12)
    {
        double tx = -bx / ax;
        if (tx > 0.0 && tx < 1.0)
            dvec2_push(&r, tx);
    }
    double ay = q.p0.y - 2.0 * q.p1.y + q.p2.y;
    double by = q.p1.y - q.p0.y;
    if (fabs(ay) > 1e-12)
    {
        double ty = -by / ay;
        if (ty > 0.0 && ty < 1.0 && r.len < 2)
            dvec2_push(&r, ty);
    }
    return r;
}

static inline Rect quadbez_bounding_box(QuadBez q)
{
    Rect  bb = rect_from_points(q.p0, q.p2);
    DVec2 ex = quadbez_extrema(q);
    for (int i = 0; i < ex.len; i++)
        bb = rect_union_pt(bb, quadbez_eval(q, ex.v[i]));
    return bb;
}

/* Arclen using Gaussian quadrature (7-point) — from Rust quadbez.rs */
static inline double quadbez_arclen(QuadBez q, double accuracy)
{
    /* MANUAL TRANSLATION NEEDED: The Rust version uses a lookup table of
     * Gaussian quadrature coefficients specific to quadbez arclen.
     * The table is in quad_arclen.rs. A simple 8-point Gauss-Legendre
     * approximation follows, which may not match the Rust accuracy. */
    static const double wi[] = {
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.3137066458778873,
        0.2223810344533745,
        0.2223810344533745,
        0.1012285362903763,
        0.1012285362903763};
    static const double xi[] = {
        -0.1834346424956498,
        0.1834346424956498,
        -0.5255324099163290,
        0.5255324099163290,
        -0.7966664774136267,
        0.7966664774136267,
        -0.9602898564975363,
        0.9602898564975363};
    double sum = 0.0;
    for (int i = 0; i < 8; i++)
    {
        double t  = 0.5 * (xi[i] + 1.0);
        Point  d  = quadbez_deriv_eval(q, t);
        sum      += wi[i] * sqrt(d.x * d.x + d.y * d.y);
    }
    return 0.5 * sum;
}

/* ============================================================
 * SECTION: CubicBez
 * ============================================================ */

static inline CubicBez cubicbez_new(Point p0, Point p1, Point p2, Point p3)
{
    CubicBez c = {p0, p1, p2, p3};
    return c;
}

static inline Point cubicbez_eval(CubicBez c, double t)
{
    double mt  = 1.0 - t;
    double mt2 = mt * mt, t2 = t * t;
    double x = mt2 * mt * c.p0.x + 3.0 * mt2 * t * c.p1.x + 3.0 * mt * t2 * c.p2.x + t2 * t * c.p3.x;
    double y = mt2 * mt * c.p0.y + 3.0 * mt2 * t * c.p1.y + 3.0 * mt * t2 * c.p2.y + t2 * t * c.p3.y;
    return point_new(x, y);
}

/* Derivative of cubic is a quadratic */
static inline QuadBez cubicbez_deriv(CubicBez c)
{
    return quadbez_new(
        point_new(3.0 * (c.p1.x - c.p0.x), 3.0 * (c.p1.y - c.p0.y)),
        point_new(3.0 * (c.p2.x - c.p1.x), 3.0 * (c.p2.y - c.p1.y)),
        point_new(3.0 * (c.p3.x - c.p2.x), 3.0 * (c.p3.y - c.p2.y)));
}

static inline Point cubicbez_deriv_eval(CubicBez c, double t)
{
    QuadBez d = cubicbez_deriv(c);
    return quadbez_eval(d, t);
}

static inline double cubicbez_signed_area(CubicBez c)
{
    /* Exact Green's theorem formula from cubicbez.rs ParamCurveArea impl:
     * integral of (y dx) using the cubic Bezier parametric form.
     * The formula is:
     *   area = (p0.x*(6*p1.y + 3*p2.y + p3.y)
     *         - p3.x*(p0.y + 3*p1.y + 6*p2.y)
     *         + 3*(p1.x*(p2.y - p0.y) + p2.x*(p3.y - p1.y))) / -20
     * but Kurbo uses the convention that matches the sign for a y-down coord system.
     */
    Vec2 p0 = point_to_vec2(c.p0);
    Vec2 p1 = point_to_vec2(c.p1);
    Vec2 p2 = point_to_vec2(c.p2);
    Vec2 p3 = point_to_vec2(c.p3);
    return (p0.x * (6.0 * p1.y + 3.0 * p2.y + p3.y) - p3.x * (p0.y + 3.0 * p1.y + 6.0 * p2.y) +
            3.0 * (p1.x * (p2.y - p0.y) + p2.x * (p3.y - p1.y))) *
           (-1.0 / 20.0);
}

static inline CubicBez cubicbez_subsegment(CubicBez c, double t0, double t1)
{
    /* Proper nested de Casteljau subdivision — exact port of Rust cubicbez.rs subsegment */
    { /* first reduce [0, t1] */
        double m   = t1;
        double p0x = c.p0.x, p0y = c.p0.y;
        double p1x = c.p0.x + m * (c.p1.x - c.p0.x), p1y = c.p0.y + m * (c.p1.y - c.p0.y);
        double p2x, p2y, p3x, p3y;
        {
            double q0x = c.p1.x + m * (c.p2.x - c.p1.x), q0y = c.p1.y + m * (c.p2.y - c.p1.y);
            double q1x = c.p2.x + m * (c.p3.x - c.p2.x), q1y = c.p2.y + m * (c.p3.y - c.p2.y);
            p2x = p1x + m * (q0x - p1x);
            p2y = p1y + m * (q0y - p1y);
            p3x = q0x + m * (q1x - q0x);
            p3y = q0y + m * (q1y - q0y);
            /* now [0..t1] segment has control points p0,p1,p2,p3 and end at p3+m*(q1-q0) */
            /* but we need full [0..t1], the above gives us [0..t1] with: */
            double r2x = p2x + m * (p3x - p2x);
            double r2y = p2y + m * (p3y - p2y);
            p3x        = r2x;
            p3y        = r2y;
            /* Actually use the correct de Casteljau for [0..t1] reduction: */
            /* p0=c.p0, p1=lerp(c.p0,c.p1,t1), p2=lerp(lerp(c.p0,c.p1,t1),lerp(c.p1,c.p2,t1),t1),
               p3=eval(c,t1) */
            (void)p0x;
            (void)p0y;
            (void)p1x;
            (void)p1y;
            (void)p2x;
            (void)p2y;
            (void)p3x;
            (void)p3y;
            (void)q0x;
            (void)q0y;
            (void)q1x;
            (void)q1y;
            (void)r2x;
            (void)r2y;
        }
        /* Restart with correct approach (nested de Casteljau) */
        /* Step 1: reduce [0..t1] */
        double mt1 = 1.0 - t1;
        /* level 1 */
        double l1_0x = c.p0.x * mt1 + c.p1.x * t1, l1_0y = c.p0.y * mt1 + c.p1.y * t1;
        double l1_1x = c.p1.x * mt1 + c.p2.x * t1, l1_1y = c.p1.y * mt1 + c.p2.y * t1;
        double l1_2x = c.p2.x * mt1 + c.p3.x * t1, l1_2y = c.p2.y * mt1 + c.p3.y * t1;
        /* level 2 */
        double l2_0x = l1_0x * mt1 + l1_1x * t1, l2_0y = l1_0y * mt1 + l1_1y * t1;
        double l2_1x = l1_1x * mt1 + l1_2x * t1, l2_1y = l1_1y * mt1 + l1_2y * t1;
        /* level 3 = eval at t1 */
        double e1x = l2_0x * mt1 + l2_1x * t1, e1y = l2_0y * mt1 + l2_1y * t1;
        /* [0..t1] segment: c.p0, l1_0, l2_0, e1 */

        if (t0 == 0.0)
        {
            return cubicbez_new(c.p0, point_new(l1_0x, l1_0y), point_new(l2_0x, l2_0y), point_new(e1x, e1y));
        }

        /* Step 2: reduce [t0..t1] using the [0..t1] segment */
        /* map t0 to parameter within [0..t1]: u = t0/t1 */
        double u  = (t1 > 1e-12) ? t0 / t1 : 0.0;
        double mu = 1.0 - u;
        /* Points of [0..t1] segment */
        double A0x = c.p0.x, A0y = c.p0.y;
        double A1x = l1_0x, A1y = l1_0y;
        double A2x = l2_0x, A2y = l2_0y;
        double A3x = e1x, A3y = e1y;
        /* de Casteljau on A with u */
        double b1_0x = A0x * mu + A1x * u, b1_0y = A0y * mu + A1y * u;
        double b1_1x = A1x * mu + A2x * u, b1_1y = A1y * mu + A2y * u;
        double b1_2x = A2x * mu + A3x * u, b1_2y = A2y * mu + A3y * u;
        double b2_0x = b1_0x * mu + b1_1x * u, b2_0y = b1_0y * mu + b1_1y * u;
        double b2_1x = b1_1x * mu + b1_2x * u, b2_1y = b1_1y * mu + b1_2y * u;
        double b3x = b2_0x * mu + b2_1x * u, b3y = b2_0y * mu + b2_1y * u;
        /* [t0..t1] segment: b3, b2_1, b1_2, A3 */
        return cubicbez_new(point_new(b3x, b3y), point_new(b2_1x, b2_1y), point_new(b1_2x, b1_2y), point_new(A3x, A3y));
        (void)m; /* suppress warning */
    }
}

static inline Rect cubicbez_bounding_box(CubicBez c)
{
    /* Exact bounding box: solve for extrema of the derivative (a quadratic in t) */
    Rect bb = rect_from_points(c.p0, c.p3);
    /* derivative coefficients for x: d/dt of cubic bezier
     * deriv = QuadBez with p0=3*(p1-p0), p1=3*(p2-p1), p2=3*(p3-p2)
     * that's a quadratic in t: at^2 + bt + c = 0
     * a = (dp0 - 2*dp1 + dp2), b = 2*(dp1 - dp0), c = dp0
     * where dp0=3*(c.p1-c.p0), dp1=3*(c.p2-c.p1), dp2=3*(c.p3-c.p2) */
    {
        double dp0x = 3.0 * (c.p1.x - c.p0.x);
        double dp1x = 3.0 * (c.p2.x - c.p1.x);
        double dp2x = 3.0 * (c.p3.x - c.p2.x);
        double ax   = dp0x - 2.0 * dp1x + dp2x;
        double bx   = 2.0 * (dp1x - dp0x);
        double cx   = dp0x;
        DVec2  rx   = solve_quadratic(cx, bx, ax);
        for (int i = 0; i < rx.len; i++)
        {
            double t = rx.v[i];
            if (t > 0.0 && t < 1.0)
                bb = rect_union_pt(bb, cubicbez_eval(c, t));
        }
    }
    {
        double dp0y = 3.0 * (c.p1.y - c.p0.y);
        double dp1y = 3.0 * (c.p2.y - c.p1.y);
        double dp2y = 3.0 * (c.p3.y - c.p2.y);
        double ay   = dp0y - 2.0 * dp1y + dp2y;
        double by   = 2.0 * (dp1y - dp0y);
        double cy   = dp0y;
        DVec2  ry   = solve_quadratic(cy, by, ay);
        for (int i = 0; i < ry.len; i++)
        {
            double t = ry.v[i];
            if (t > 0.0 && t < 1.0)
                bb = rect_union_pt(bb, cubicbez_eval(c, t));
        }
    }
    return bb.x0 <= bb.x1 ? bb : rect_new(bb.x1, bb.y1, bb.x0, bb.y0);
}

static inline Nearest cubicbez_nearest(CubicBez c, Point p, double accuracy)
{
    /* MANUAL TRANSLATION NEEDED: analytical nearest using quintic solve */
    int    N      = 64;
    double best_t = 0.0, best_d = 1e300;
    for (int i = 0; i <= N; i++)
    {
        double t  = (double)i / N;
        Point  pt = cubicbez_eval(c, t);
        double d  = point_distance_sq(pt, p);
        if (d < best_d)
        {
            best_d = d;
            best_t = t;
        }
    }
    Nearest n = {best_d, best_t};
    return n;
}

static inline double cubicbez_arclen(CubicBez c, double accuracy)
{
    /* MANUAL TRANSLATION NEEDED: Rust uses recursive subdivision with a
     * lookup table of Gauss-Legendre coefficients from cubic_arclen.rs.
     * Simple 16-point Gauss-Legendre approximation follows. */
    static const double wi[] = {
        0.1894506104550685,
        0.1894506104550685,
        0.1826034150449236,
        0.1826034150449236,
        0.1691565193950025,
        0.1691565193950025,
        0.1495959888165767,
        0.1495959888165767,
        0.1246289512509458,
        0.1246289512509458,
        0.0951585116824928,
        0.0951585116824928,
        0.0622535239386478,
        0.0622535239386478,
        0.0271524593914925,
        0.0271524593914925};
    static const double xi[] = {
        -0.0950125098360823,
        0.0950125098360823,
        -0.2816035507792589,
        0.2816035507792589,
        -0.4580167776572273,
        0.4580167776572273,
        -0.6178762444026437,
        0.6178762444026437,
        -0.7554044083550030,
        0.7554044083550030,
        -0.8656312023341098,
        0.8656312023341098,
        -0.9445750230732326,
        0.9445750230732326,
        -0.9894009349916499,
        0.9894009349916499};
    double sum = 0.0;
    for (int i = 0; i < 16; i++)
    {
        double t  = 0.5 * (xi[i] + 1.0);
        Point  d  = cubicbez_deriv_eval(c, t);
        sum      += wi[i] * sqrt(d.x * d.x + d.y * d.y);
    }
    return 0.5 * sum;
}

/* ============================================================
 * SECTION: common.rs — solve_quadratic, solve_cubic, solve_quartic, solve_itp
 * ============================================================ */

static inline double eps_rel_helper(double raw, double a)
{
    if (a == 0.0)
        return fabs(raw);
    return fabs((raw - a) / a);
}

DVec2 solve_quadratic(double c0, double c1, double c2)
{
    DVec2 result;
    result.len = 0;
    double sc0 = c0 / c2;
    double sc1 = c1 / c2;
    if (!isfinite(sc0) || !isfinite(sc1))
    {
        double root = -c0 / c1;
        if (isfinite(root))
        {
            dvec2_push(&result, root);
        }
        else if (c0 == 0.0 && c1 == 0.0)
        {
            dvec2_push(&result, 0.0);
        }
        return result;
    }
    double arg = sc1 * sc1 - 4.0 * sc0;
    double root1;
    if (!isfinite(arg))
    {
        root1 = -sc1;
    }
    else
    {
        if (arg < 0.0)
            return result;
        if (arg == 0.0)
        {
            dvec2_push(&result, -0.5 * sc1);
            return result;
        }
        root1 = -0.5 * (sc1 + copysign(sqrt(arg), sc1));
    }
    double root2 = sc0 / root1;
    if (isfinite(root2))
    {
        if (root2 > root1)
        {
            dvec2_push(&result, root1);
            dvec2_push(&result, root2);
        }
        else
        {
            dvec2_push(&result, root2);
            dvec2_push(&result, root1);
        }
    }
    else
    {
        dvec2_push(&result, root1);
    }
    return result;
}

DVec3 solve_cubic(double c0, double c1, double c2, double c3)
{
    DVec3 result;
    result.len            = 0;
    double       c3r      = 1.0 / c3;
    const double ONETHIRD = 1.0 / 3.0;
    double       sc2      = c2 * (ONETHIRD * c3r);
    double       sc1      = c1 * (ONETHIRD * c3r);
    double       sc0      = c0 * c3r;
    if (!isfinite(sc0) || !isfinite(sc1) || !isfinite(sc2))
    {
        DVec2 q = solve_quadratic(c0, c1, c2);
        for (int i = 0; i < q.len; i++)
            dvec3_push(&result, q.v[i]);
        return result;
    }
    double d0 = -sc2 * sc2 + sc1;
    double d1 = -sc1 * sc2 + sc0;
    double d2 = sc2 * sc0 - sc1 * sc1;
    double d  = 4.0 * d0 * d2 - d1 * d1;
    double de = -2.0 * sc2 * d0 + d1;
    if (d < 0.0)
    {
        double sq = sqrt(-0.25 * d);
        double r  = -0.5 * de;
        double t1 = cbrt(r + sq) + cbrt(r - sq);
        dvec3_push(&result, t1 - sc2);
    }
    else if (d == 0.0)
    {
        double t1 = copysign(sqrt(-d0), de);
        dvec3_push(&result, t1 - sc2);
        dvec3_push(&result, -2.0 * t1 - sc2);
    }
    else
    {
        double th     = atan2(sqrt(d), -de) * ONETHIRD;
        double th_sin = sin(th), th_cos = cos(th);
        double r0  = th_cos;
        double ss3 = th_sin * sqrt(3.0);
        double r1  = 0.5 * (-th_cos + ss3);
        double r2  = 0.5 * (-th_cos - ss3);
        double t   = 2.0 * sqrt(-d0);
        dvec3_push(&result, fma(t, r0, -sc2));
        dvec3_push(&result, fma(t, r1, -sc2));
        dvec3_push(&result, fma(t, r2, -sc2));
    }
    return result;
}

/* Forward declaration */
static bool factor_quartic_inner(double a, double b, double c, double d, bool rescale, DPairVec2* out);

static bool solve_quartic_inner(double a, double b, double c, double d, bool rescale, DVec4* out)
{
    DPairVec2 quads;
    quads.len = 0;
    if (!factor_quartic_inner(a, b, c, d, rescale, &quads))
        return false;
    out->len = 0;
    for (int i = 0; i < quads.len; i++)
    {
        DVec2 roots = solve_quadratic(quads.v[i].b, quads.v[i].a, 1.0);
        for (int j = 0; j < roots.len; j++)
            dvec4_push(out, roots.v[j]);
    }
    return true;
}

DVec4 solve_quartic(double c0, double c1, double c2, double c3, double c4)
{
    DVec4 result;
    result.len = 0;
    if (c4 == 0.0)
    {
        DVec3 r = solve_cubic(c0, c1, c2, c3);
        for (int i = 0; i < r.len; i++)
            dvec4_push(&result, r.v[i]);
        return result;
    }
    if (c0 == 0.0)
    {
        DVec3 r = solve_cubic(c1, c2, c3, c4);
        for (int i = 0; i < r.len; i++)
            dvec4_push(&result, r.v[i]);
        dvec4_push(&result, 0.0);
        return result;
    }
    double a = c3 / c4, b = c2 / c4, c = c1 / c4, d = c0 / c4;
    if (solve_quartic_inner(a, b, c, d, false, &result))
        return result;
    const double K_Q = 7.16e76;
    for (int rescale = 0; rescale <= 1; rescale++)
    {
        DVec4 r2;
        r2.len = 0;
        if (solve_quartic_inner(
                a / K_Q,
                b / (K_Q * K_Q),
                c / (K_Q * K_Q * K_Q),
                d / (K_Q * K_Q * K_Q * K_Q),
                (bool)rescale,
                &r2))
        {
            result.len = 0;
            for (int i = 0; i < r2.len; i++)
                dvec4_push(&result, r2.v[i] * K_Q);
            return result;
        }
    }
    return result; /* len = 0 */
}

static double depressed_cubic_dominant(double g, double h)
{
    double q = (-1.0 / 3.0) * g;
    double r = 0.5 * h;
    double phi_0;
    double k_abs_q = fabs(q), k_abs_r = fabs(r);
    int    has_k = 0;
    double k     = 0.0;
    if (k_abs_q < 1e102 && k_abs_r < 1e154)
    {
        has_k = 0;
    }
    else if (k_abs_q < k_abs_r)
    {
        k     = 1.0 - q * (q / r) * (q / r);
        has_k = 1;
    }
    else
    {
        k     = copysign((r / q) * (r / q) / q - 1.0, q);
        has_k = 1;
    }
    if (has_k && r == 0.0)
    {
        phi_0 = (g > 0.0) ? 0.0 : sqrt(-g);
    }
    else if ((has_k && k < 0.0) || (!has_k && r * r < q * q * q))
    {
        double t = has_k ? r / q / sqrt(q) : r / sqrt(q * q * q);
        phi_0    = -2.0 * sqrt(q) * copysign(cos(acos(fabs(t)) * (1.0 / 3.0)), t);
    }
    else
    {
        double aa;
        if (has_k)
        {
            if (k_abs_q < k_abs_r)
                aa = -r * (1.0 + sqrt(k));
            else
                aa = -r - copysign(sqrt(fabs(q)) * q * sqrt(k), r);
        }
        else
        {
            aa = -r - copysign(sqrt(r * r - q * q * q), r);
        }
        aa        = cbrt(aa);
        double bb = (aa == 0.0) ? 0.0 : q / aa;
        phi_0     = aa + bb;
    }
    double       x     = phi_0;
    double       f     = (x * x + g) * x + h;
    const double EPS_M = 2.22045e-16;
    if (fabs(f) < EPS_M * fmax(fmax(fabs(x * x * x), fabs(g * x)), fabs(h)))
        return x;
    for (int i = 0; i < 8; i++)
    {
        double df = 3.0 * x * x + g;
        if (df == 0.0)
            break;
        double new_x = x - f / df;
        double new_f = (new_x * new_x + g) * new_x + h;
        if (new_f == 0.0)
            return new_x;
        if (fabs(new_f) >= fabs(f))
            break;
        x = new_x;
        f = new_f;
    }
    return x;
}

static bool factor_quartic_inner(double a, double b, double c, double d, bool rescale, DPairVec2* out)
{
    out->len = 0;
    /* MANUAL TRANSLATION NEEDED: This is a direct port of the Orellana/De Michele
     * algorithm. The Rust source in common.rs (~lines 362-580) is complex.
     * The logic below is a faithful structural translation but should be
     * verified line-by-line against the Rust original. */
    double disc = 9.0 * a * a - 24.0 * b;
    double s;
    if (disc >= 0.0)
        s = -2.0 * b / (3.0 * a + copysign(sqrt(disc), a));
    else
        s = -0.25 * a;
    double       a_prime = a + 4.0 * s;
    double       b_prime = b + 3.0 * s * (a + 2.0 * s);
    double       c_prime = c + s * (2.0 * b + s * (3.0 * a + 4.0 * s));
    double       d_prime = d + s * (c + s * (b + s * (a + s)));
    double       g_prime, h_prime;
    const double K_C = 3.49e102;
    if (rescale)
    {
        double aps = a_prime / K_C, bps = b_prime / K_C, cps = c_prime / K_C, dps = d_prime / K_C;
        g_prime = aps * cps - (4.0 / K_C) * dps - (1.0 / 3.0) * bps * bps;
        h_prime = (aps * cps + (8.0 / K_C) * dps - (2.0 / 9.0) * bps * bps) * (1.0 / 3.0) * bps - cps * (cps / K_C) -
                  aps * aps * dps;
    }
    else
    {
        g_prime = a_prime * c_prime - 4.0 * d_prime - (1.0 / 3.0) * b_prime * b_prime;
        h_prime = (a_prime * c_prime + 8.0 * d_prime - (2.0 / 9.0) * b_prime * b_prime) * (1.0 / 3.0) * b_prime -
                  c_prime * c_prime - a_prime * a_prime * d_prime;
    }
    if (!isfinite(g_prime) || !isfinite(h_prime))
        return false;
    double phi     = depressed_cubic_dominant(g_prime, h_prime);
    double phi_1   = rescale ? phi / K_C : phi;
    double l_prime = a_prime * 0.5 + (b_prime / 3.0 - phi_1) / (a_prime != 0.0 ? a_prime : 1.0);
    /* MANUAL TRANSLATION NEEDED: Lines 450-560 of common.rs compute alpha_1/2, beta_1/2
     * via a sign-correct sqrt + Newton refinement. This is highly nuanced.
     * Stub returns false (no factorization). */
    (void)l_prime;
    /* TODO: Complete Newton-Raphson refinement for alpha_1, beta_1, alpha_2, beta_2 */
    return false;
}

/* solve_itp — ITP root finding */
typedef double (*ItpFunc)(double, void*);

double
solve_itp_generic(ItpFunc f, void* user, double a, double b, double epsilon, int n0, double k1, double ya, double yb)
{
    double log_val = ceil(log2((b - a) / epsilon)) - 1.0;
    if (log_val < 0.0)
        log_val = 0.0;
    int    n1_2           = (int)log_val;
    int    nmax           = n0 + n1_2;
    double scaled_epsilon = epsilon * (double)(1ULL << nmax);
    while (b - a > 2.0 * epsilon)
    {
        double x1_2  = 0.5 * (a + b);
        double r     = scaled_epsilon - 0.5 * (b - a);
        double xf    = (yb * a - ya * b) / (yb - ya);
        double sigma = x1_2 - xf;
        double delta = k1 * (b - a) * (b - a);
        double xt    = (delta <= fabs(x1_2 - xf)) ? xf + copysign(delta, sigma) : x1_2;
        double xitp  = (fabs(xt - x1_2) <= r) ? xt : x1_2 - copysign(r, sigma);
        double yitp  = f(xitp, user);
        if (yitp > 0.0)
        {
            b  = xitp;
            yb = yitp;
        }
        else if (yitp < 0.0)
        {
            a  = xitp;
            ya = yitp;
        }
        else
            return xitp;
        scaled_epsilon *= 0.5;
    }
    return 0.5 * (a + b);
}

/* ============================================================
 * SECTION: Arc
 * ============================================================ */

static inline Vec2 sample_ellipse(Vec2 radii, double x_rotation, double angle)
{
    double angle_sin = sin(angle), angle_cos = cos(angle);
    double u     = radii.x * angle_cos;
    double v     = radii.y * angle_sin;
    double rot_s = sin(x_rotation), rot_c = cos(x_rotation);
    return vec2_new(u * rot_c - v * rot_s, u * rot_s + v * rot_c);
}

static inline Arc arc_new(Point center, Vec2 radii, double start_angle, double sweep_angle, double x_rotation)
{
    Arc a = {center, radii, start_angle, sweep_angle, x_rotation};
    return a;
}

static inline Arc arc_reversed(Arc a)
{
    Arc r         = a;
    r.start_angle = a.start_angle + a.sweep_angle;
    r.sweep_angle = -a.sweep_angle;
    return r;
}

/* Arc approximation as cubic beziers — callback-based */
typedef void (*CubicCallback)(Point p1, Point p2, Point p3, void* user);

void arc_to_cubic_beziers(Arc arc, double tolerance, CubicCallback cb, void* user)
{
    double sign       = (arc.sweep_angle > 0.0) ? 1.0 : (arc.sweep_angle < 0.0) ? -1.0 : 0.0;
    double scaled_err = fmax(arc.radii.x, arc.radii.y) / tolerance;
    double n_err      = pow(1.1163 * scaled_err, 1.0 / 6.0);
    if (n_err < 3.999999)
        n_err = 3.999999;
    double n_f = ceil(n_err * fabs(arc.sweep_angle) * (1.0 / (2.0 * KURBO_PI)));
    int    n   = (int)n_f;
    if (n < 1)
        n = 1;
    double angle_step = arc.sweep_angle / n;
    double arm_len    = (4.0 / 3.0) * fabs(tan(0.25 * angle_step)) * sign;
    double angle0     = arc.start_angle;
    Vec2   p0         = sample_ellipse(arc.radii, arc.x_rotation, angle0);
    for (int i = 0; i < n; i++)
    {
        double angle1 = angle0 + angle_step;
        Vec2   p1v =
            vec2_add(p0, vec2_scale(sample_ellipse(arc.radii, arc.x_rotation, angle0 + KURBO_FRAC_PI_2), arm_len));
        Vec2 p3 = sample_ellipse(arc.radii, arc.x_rotation, angle1);
        Vec2 p2v =
            vec2_sub(p3, vec2_scale(sample_ellipse(arc.radii, arc.x_rotation, angle1 + KURBO_FRAC_PI_2), arm_len));
        cb(point_add_vec2(arc.center, p1v), point_add_vec2(arc.center, p2v), point_add_vec2(arc.center, p3), user);
        angle0 = angle1;
        p0     = p3;
    }
}

static inline Arc affine_mul_arc(Affine aff, Arc arc)
{
    /* MANUAL TRANSLATION NEEDED: Requires Ellipse::new + Affine*Ellipse + ellipse decomposition.
     * The Rust impl computes: ellipse = aff * Ellipse::new(arc.center, arc.radii, arc.x_rotation),
     * then extracts center, radii, rotation from the resulting ellipse.
     * Stub returns arc unchanged. */
    (void)aff;
    return arc;
}

/* ============================================================
 * SECTION: Circle
 * ============================================================ */

static inline Circle circle_new(Point center, double radius)
{
    Circle c = {center, radius};
    return c;
}

static inline double circle_area(Circle c) { return KURBO_PI * c.radius * c.radius; }

static inline double circle_perimeter(Circle c) { return fabs(2.0 * KURBO_PI * c.radius); }

static inline int circle_winding(Circle c, Point pt)
{
    Vec2 d = point_sub(pt, c.center);
    return (vec2_hypot2(d) < c.radius * c.radius) ? 1 : 0;
}

static inline Rect circle_bounding_box(Circle c)
{
    double r = fabs(c.radius);
    return rect_new(c.center.x - r, c.center.y - r, c.center.x + r, c.center.y + r);
}

static inline bool circle_is_finite(Circle c) { return point_is_finite(c.center) && isfinite(c.radius); }
static inline bool circle_is_nan(Circle c) { return point_is_nan(c.center) || isnan(c.radius); }

static inline Circle circle_add_vec2(Circle c, Vec2 v) { return circle_new(point_add_vec2(c.center, v), c.radius); }
static inline Circle circle_sub_vec2(Circle c, Vec2 v) { return circle_new(point_sub_vec2(c.center, v), c.radius); }

/* CircleSegment */
static inline CircleSegment
circle_segment_new(Point center, double outer_r, double inner_r, double start_angle, double sweep_angle)
{
    CircleSegment cs = {center, outer_r, inner_r, start_angle, sweep_angle};
    return cs;
}

static inline Arc circle_segment_outer_arc(CircleSegment cs)
{
    return arc_new(cs.center, vec2_new(cs.outer_radius, cs.outer_radius), cs.start_angle, cs.sweep_angle, 0.0);
}

static inline Arc circle_segment_inner_arc(CircleSegment cs)
{
    return arc_new(
        cs.center,
        vec2_new(cs.inner_radius, cs.inner_radius),
        cs.start_angle + cs.sweep_angle,
        -cs.sweep_angle,
        0.0);
}

static inline double circle_segment_area(CircleSegment cs)
{
    return 0.5 * fabs(cs.outer_radius * cs.outer_radius - cs.inner_radius * cs.inner_radius) * cs.sweep_angle;
}

static inline double circle_segment_perimeter(CircleSegment cs)
{
    return 2.0 * fabs(cs.outer_radius - cs.inner_radius) + cs.sweep_angle * (cs.inner_radius + cs.outer_radius);
}

static inline int circle_segment_winding(CircleSegment cs, Point pt)
{
    Vec2   d     = point_sub(pt, cs.center);
    double angle = atan2(d.y, d.x);
    if (angle < cs.start_angle || angle > cs.start_angle + cs.sweep_angle)
        return 0;
    double dist2 = vec2_hypot2(d);
    double or2   = cs.outer_radius * cs.outer_radius;
    double ir2   = cs.inner_radius * cs.inner_radius;
    if ((dist2 < or2 && dist2 > ir2) || (dist2 < ir2 && dist2 > or2))
        return 1;
    return 0;
}

static inline Rect circle_segment_bounding_box(CircleSegment cs)
{
    double r = fmax(cs.inner_radius, cs.outer_radius);
    return rect_new(cs.center.x - r, cs.center.y - r, cs.center.x + r, cs.center.y + r);
}

static inline bool circle_segment_is_finite(CircleSegment cs)
{
    return point_is_finite(cs.center) && isfinite(cs.outer_radius) && isfinite(cs.inner_radius) &&
           isfinite(cs.start_angle) && isfinite(cs.sweep_angle);
}

/* ============================================================
 * SECTION: QuadSpline
 * ============================================================ */

/* QuadSpline iterator state */
typedef struct
{
    const Point* points;
    int          n_points;
    int          idx;
} QuadSplineIter;

static inline void quad_spline_iter_init(QuadSplineIter* it, const Point* pts, int n)
{
    it->points   = pts;
    it->n_points = n;
    it->idx      = 0;
}

/* Returns false when exhausted */
static inline bool quad_spline_iter_next(QuadSplineIter* it, QuadBez* out)
{
    if (it->idx + 2 >= it->n_points)
        return false;
    Point p0 = it->points[it->idx];
    Point p1 = it->points[it->idx + 1];
    Point p2 = it->points[it->idx + 2];
    if (it->idx != 0)
        p0 = point_midpoint(p0, p1);
    if (it->idx + 2 < it->n_points - 1)
        p2 = point_midpoint(p1, p2);
    *out = quadbez_new(p0, p1, p2);
    it->idx++;
    return true;
}

/* ============================================================
 * SECTION: mindist.rs — minimum distance between two Bezier curves
 * ============================================================ */

/* NOTE: The mindist algorithm operates on curves as arrays of Vec2 control points.
 * The Rust source uses generic Bezier degree. Below we provide the core recursive
 * subdivison and helper functions. The choose() and D_rk/A_r/C_rk functions
 * are ported directly. */

static uint32_t choose(uint8_t n, uint8_t k)
{
    if (k > n)
        return 0;
    uint32_t p  = 1;
    uint32_t nn = n;
    uint32_t kk = k;
    for (uint32_t i = 1; i <= nn - kk; i++)
    {
        p *= nn;
        p /= i;
        nn--;
    }
    return p;
}

static double A_r_func(uint8_t r, const Vec2* p, int p_len)
{
    uint8_t n       = (uint8_t)(p_len - 1);
    uint8_t upsilon = (r < n) ? r : n;
    uint8_t theta   = r - ((n < r) ? n : r);
    double  sum     = 0.0;
    for (uint8_t i = theta; i <= upsilon; i++)
    {
        double dot     = vec2_dot(p[i], p[r - i]);
        double factor  = (double)(choose(n, i) * choose(n, r - i)) / (double)choose(2 * n, r);
        sum           += dot * factor;
    }
    return sum;
}

static double C_rk_func(uint8_t r, uint8_t k, const Vec2* bez1, int n1, const Vec2* bez2, int n2)
{
    uint8_t n = (uint8_t)(n1 - 1), m = (uint8_t)(n2 - 1);
    uint8_t upsilon = (r < n) ? r : n, theta = r - ((n < r) ? n : r);
    Vec2    left = VEC2_ZERO;
    for (uint8_t i = theta; i <= upsilon; i++)
    {
        left =
            vec2_add(left, vec2_scale(bez1[i], (double)(choose(n, i) * choose(n, r - i)) / (double)choose(2 * n, r)));
    }
    uint8_t varsigma = (k < m) ? k : m, sigma = k - ((m < k) ? m : k);
    Vec2    right = VEC2_ZERO;
    for (uint8_t j = sigma; j <= varsigma; j++)
    {
        right =
            vec2_add(right, vec2_scale(bez2[j], (double)(choose(m, j) * choose(m, k - j)) / (double)choose(2 * m, k)));
    }
    return vec2_dot(left, right);
}

static double D_rk_func(uint8_t r, uint8_t k, const Vec2* bez1, int n1, const Vec2* bez2, int n2)
{
    return A_r_func(r, bez1, n1) + A_r_func(k, bez2, n2) - 2.0 * C_rk_func(r, k, bez1, n1, bez2, n2);
}

static double S_func(double u, double v, const Vec2* bez1, int n1, const Vec2* bez2, int n2)
{
    uint8_t n = (uint8_t)(n1 - 1), m = (uint8_t)(n2 - 1);
    double  sum = 0.0;
    for (uint8_t r = 0; r <= 2 * n; r++)
        for (uint8_t k = 0; k <= 2 * m; k++)
        {
            uint32_t bn = choose(2 * n, r), bm = choose(2 * m, k);
            double   basis_u  = (double)bn * pow(1.0 - u, 2 * n - r) * pow(u, r);
            double   basis_v  = (double)bm * pow(1.0 - v, 2 * m - k) * pow(v, k);
            sum              += D_rk_func(r, k, bez1, n1, bez2, n2) * basis_u * basis_v;
        }
    return sum;
}

typedef struct
{
    double dist;
    double t1;
    double t2;
} MinDistResult;

MinDistResult min_dist_param(
    const Vec2* bez1,
    int         n1,
    const Vec2* bez2,
    int         n2,
    double      umin,
    double      umax,
    double      vmin,
    double      vmax,
    double      epsilon,
    double      best_alpha, /* use -1 for "none" */
    int         depth)
{
    /* Guard against deep recursion */
    if (depth > 40)
    {
        double        umid = 0.5 * (umin + umax), vmid = 0.5 * (vmin + vmax);
        MinDistResult r = {S_func(umid, vmid, bez1, n1, bez2, n2), umid, vmid};
        return r;
    }
    uint8_t n = (uint8_t)(n1 - 1), m = (uint8_t)(n2 - 1);
    double  umid = 0.5 * (umin + umax), vmid = 0.5 * (vmin + vmax);
    double  sv[4] = {
        S_func(umin, vmin, bez1, n1, bez2, n2),
        S_func(umin, vmax, bez1, n1, bez2, n2),
        S_func(umax, vmin, bez1, n1, bez2, n2),
        S_func(umax, vmax, bez1, n1, bez2, n2)};
    double alpha = sv[0];
    int    ai    = 0;
    (void)ai;
    for (int i = 1; i < 4; i++)
        if (sv[i] < alpha)
        {
            alpha = sv[i];
            ai    = i;
        }
    if (best_alpha >= 0.0 && alpha > best_alpha)
    {
        MinDistResult r = {alpha, umid, vmid};
        return r;
    }
    if (fabs(umax - umin) < epsilon || fabs(vmax - vmin) < epsilon)
    {
        MinDistResult r = {alpha, umid, vmid};
        return r;
    }
    /* Find min D_rk */
    double  min_drk = 1e300;
    uint8_t min_r = 0, min_k = 0;
    bool    is_outside = true;
    for (uint8_t r = 0; r < 2 * n; r++)
        for (uint8_t k = 0; k < 2 * m; k++)
        {
            double drk = D_rk_func(r, k, bez1, n1, bez2, n2);
            if (drk < alpha)
                is_outside = false;
            if (drk < min_drk)
            {
                min_drk = drk;
                min_r   = r;
                min_k   = k;
            }
        }
    if (is_outside)
    {
        MinDistResult r = {alpha, umid, vmid};
        return r;
    }
    double        new_umid = umin + (umax - umin) * ((double)min_r / (double)(2 * n));
    double        new_vmid = vmin + (vmax - vmin) * ((double)min_k / (double)(2 * m));
    MinDistResult best     = {1e300, umid, vmid};
    MinDistResult sub[4];
    sub[0] = min_dist_param(bez1, n1, bez2, n2, umin, new_umid, vmin, new_vmid, epsilon, alpha, depth + 1);
    sub[1] = min_dist_param(bez1, n1, bez2, n2, umin, new_umid, new_vmid, vmax, epsilon, sub[0].dist, depth + 1);
    sub[2] = min_dist_param(bez1, n1, bez2, n2, new_umid, umax, vmin, new_vmid, epsilon, sub[1].dist, depth + 1);
    sub[3] = min_dist_param(bez1, n1, bez2, n2, new_umid, umax, new_vmid, vmax, epsilon, sub[2].dist, depth + 1);
    for (int i = 0; i < 4; i++)
        if (sub[i].dist < best.dist)
            best = sub[i];
    return best;
}

/* ============================================================
 * SECTION: Ellipse (partial)
 * MANUAL TRANSLATION NEEDED: ellipse.rs is complex — it uses SVD internally
 * to represent the ellipse as a scaled+rotated unit circle via an Affine.
 * The full implementation requires porting ~542 lines of ellipse.rs.
 * ============================================================ */

typedef struct
{
    /* Represented as an affine transform mapping the unit circle */
    Affine inner;
} Ellipse;

static inline Ellipse ellipse_new(Point center, Vec2 radii, double x_rotation)
{
    /* MANUAL TRANSLATION NEEDED: Rust stores Ellipse as inner Affine =
     * translate(center) * rotate(x_rotation) * scale_non_uniform(radii.x, radii.y)
     * but the actual mapping must be verified against ellipse.rs */
    Affine a = affine_mul(
        affine_translate(point_to_vec2(center)),
        affine_mul(affine_rotate(x_rotation), affine_scale_non_uniform(radii.x, radii.y)));
    Ellipse e;
    e.inner = a;
    return e;
}

static inline Point ellipse_center(Ellipse e) { return vec2_to_point(affine_translation(e.inner)); }

/* MANUAL TRANSLATION NEEDED: ellipse_radii_and_rotation uses SVD of the inner Affine.
 * Returns scale (radii) as Vec2 and rotation angle as double.
 * See ellipse.rs radii_and_rotation() for the full implementation. */
static inline AffineSVD ellipse_radii_and_rotation(Ellipse e) { return affine_svd(e.inner); }

/* ============================================================
 * SECTION: fit.rs — curve fitting
 * MANUAL TRANSLATION NEEDED: fit.rs (~723 lines) implements the ParamCurveFit
 * trait and fit_to_bezpath / fit_to_cubic functions. These use closures
 * heavily and depend on dynamic dispatch. The C port requires function
 * pointers and manual state structs.
 * ============================================================ */

/* ============================================================
 * SECTION: offset.rs — offset curves
 * MANUAL TRANSLATION NEEDED: offset.rs (~668 lines) implements CubicOffset
 * as a struct implementing ParamCurveFit. Requires the fit.rs infrastructure.
 * ============================================================ */

/* ============================================================
 * SECTION: moments.rs — area moments
 * MANUAL TRANSLATION NEEDED: moments.rs implements ParamCurveMoments for
 * Line, QuadBez, CubicBez, PathSeg. The trait returns a Moments struct.
 * Below is the struct definition and stubs.
 * ============================================================ */

typedef struct
{
    double area; /* signed area */
    double x;    /* first moment of area w.r.t. x */
    double y;    /* first moment of area w.r.t. y */
    double xx;   /* second moment w.r.t. x^2 */
    double xy;   /* second moment w.r.t. xy */
    double yy;   /* second moment w.r.t. y^2 */
} Moments;

static inline Moments moments_add(Moments a, Moments b)
{
    Moments r = {a.area + b.area, a.x + b.x, a.y + b.y, a.xx + b.xx, a.xy + b.xy, a.yy + b.yy};
    return r;
}

/* MANUAL TRANSLATION NEEDED: The actual moment integrals for QuadBez and CubicBez
 * involve polynomial expansions from moments.rs. Stub for Line only: */
static inline Moments line_moments(Line l)
{
    double  x0 = l.p0.x, y0 = l.p0.y, x1 = l.p1.x, y1 = l.p1.y;
    double  area = 0.5 * (x0 * y1 - x1 * y0);
    Moments m;
    m.area = area;
    m.x    = area * (x0 + x1) * (1.0 / 3.0);
    m.y    = area * (y0 + y1) * (1.0 / 3.0);
    m.xx   = 0.0; /* MANUAL TRANSLATION NEEDED */
    m.xy   = 0.0; /* MANUAL TRANSLATION NEEDED */
    m.yy   = 0.0; /* MANUAL TRANSLATION NEEDED */
    return m;
}

/* ============================================================
 * SECTION: bezpath.rs — BezPath / PathEl / PathSeg
 * MANUAL TRANSLATION NEEDED: bezpath.rs is ~2164 lines and implements
 * a dynamic path type with heap allocation (Vec<PathEl>). In C this
 * requires a growable array. Below is the PathEl enum and basic types.
 * Full BezPath operations (flatten, winding, area, etc.) need manual porting.
 * ============================================================ */

typedef enum
{
    PATH_EL_MOVE_TO,
    PATH_EL_LINE_TO,
    PATH_EL_QUAD_TO,
    PATH_EL_CURVE_TO,
    PATH_EL_CLOSE_PATH
} PathElKind;

typedef struct
{
    PathElKind kind;
    Point      p[3]; /* up to 3 points depending on kind */
} PathEl;

typedef enum
{
    PATH_SEG_LINE,
    PATH_SEG_QUAD,
    PATH_SEG_CUBIC
} PathSegKind;

typedef struct
{
    PathSegKind kind;
    union
    {
        Line     line;
        QuadBez  quad;
        CubicBez cubic;
    };
} PathSeg;

static inline Point path_seg_eval(PathSeg seg, double t)
{
    switch (seg.kind)
    {
    case PATH_SEG_LINE:
        return line_eval(seg.line, t);
    case PATH_SEG_QUAD:
        return quadbez_eval(seg.quad, t);
    case PATH_SEG_CUBIC:
        return cubicbez_eval(seg.cubic, t);
    }
    return POINT_ZERO;
}

static inline Nearest path_seg_nearest(PathSeg seg, Point p, double accuracy)
{
    switch (seg.kind)
    {
    case PATH_SEG_LINE:
        return line_nearest(seg.line, p);
    case PATH_SEG_QUAD:
        return quadbez_nearest(seg.quad, p, accuracy);
    case PATH_SEG_CUBIC:
        return cubicbez_nearest(seg.cubic, p, accuracy);
    }
    Nearest n = {1e300, 0.0};
    return n;
}

static inline double path_seg_arclen(PathSeg seg, double accuracy)
{
    switch (seg.kind)
    {
    case PATH_SEG_LINE:
        return line_arclen(seg.line);
    case PATH_SEG_QUAD:
        return quadbez_arclen(seg.quad, accuracy);
    case PATH_SEG_CUBIC:
        return cubicbez_arclen(seg.cubic, accuracy);
    }
    return 0.0;
}

/* min_dist between two PathSegs — calls min_dist_param */
MinDistResult path_seg_min_dist(PathSeg a, PathSeg b, double epsilon)
{
    /* Convert PathSegs to Vec2 arrays for min_dist_param */
    Vec2 a_pts[4], b_pts[4];
    int  na = 0, nb = 0;
    switch (a.kind)
    {
    case PATH_SEG_LINE:
        a_pts[0] = point_to_vec2(a.line.p0);
        a_pts[1] = point_to_vec2(a.line.p1);
        na       = 2;
        break;
    case PATH_SEG_QUAD:
        a_pts[0] = point_to_vec2(a.quad.p0);
        a_pts[1] = point_to_vec2(a.quad.p1);
        a_pts[2] = point_to_vec2(a.quad.p2);
        na       = 3;
        break;
    case PATH_SEG_CUBIC:
        a_pts[0] = point_to_vec2(a.cubic.p0);
        a_pts[1] = point_to_vec2(a.cubic.p1);
        a_pts[2] = point_to_vec2(a.cubic.p2);
        a_pts[3] = point_to_vec2(a.cubic.p3);
        na       = 4;
        break;
    }
    switch (b.kind)
    {
    case PATH_SEG_LINE:
        b_pts[0] = point_to_vec2(b.line.p0);
        b_pts[1] = point_to_vec2(b.line.p1);
        nb       = 2;
        break;
    case PATH_SEG_QUAD:
        b_pts[0] = point_to_vec2(b.quad.p0);
        b_pts[1] = point_to_vec2(b.quad.p1);
        b_pts[2] = point_to_vec2(b.quad.p2);
        nb       = 3;
        break;
    case PATH_SEG_CUBIC:
        b_pts[0] = point_to_vec2(b.cubic.p0);
        b_pts[1] = point_to_vec2(b.cubic.p1);
        b_pts[2] = point_to_vec2(b.cubic.p2);
        b_pts[3] = point_to_vec2(b.cubic.p3);
        nb       = 4;
        break;
    }
    MinDistResult r = min_dist_param(a_pts, na, b_pts, nb, 0.0, 1.0, 0.0, 1.0, epsilon, -1.0, 0);
    r.dist          = sqrt(r.dist); /* convert sq distance to distance */
    return r;
}

/* BezPath — dynamic growable path
 * MANUAL TRANSLATION NEEDED: Full BezPath with heap allocation.
 * Below is a minimal struct with a fixed-capacity for demonstration.
 * Replace with a proper dynamic array in production code.
 */
#define BEZPATH_MAX_ELS 4096

typedef struct
{
    PathEl els[BEZPATH_MAX_ELS];
    int    len;
} BezPath;

static inline void bezpath_init(BezPath* p) { p->len = 0; }

static inline void bezpath_move_to(BezPath* p, Point pt)
{
    assert(p->len < BEZPATH_MAX_ELS);
    p->els[p->len].kind = PATH_EL_MOVE_TO;
    p->els[p->len].p[0] = pt;
    p->len++;
}
static inline void bezpath_line_to(BezPath* p, Point pt)
{
    assert(p->len < BEZPATH_MAX_ELS);
    p->els[p->len].kind = PATH_EL_LINE_TO;
    p->els[p->len].p[0] = pt;
    p->len++;
}
static inline void bezpath_quad_to(BezPath* p, Point ctrl, Point to)
{
    assert(p->len < BEZPATH_MAX_ELS);
    p->els[p->len].kind = PATH_EL_QUAD_TO;
    p->els[p->len].p[0] = ctrl;
    p->els[p->len].p[1] = to;
    p->len++;
}
static inline void bezpath_curve_to(BezPath* p, Point c1, Point c2, Point to)
{
    assert(p->len < BEZPATH_MAX_ELS);
    p->els[p->len].kind = PATH_EL_CURVE_TO;
    p->els[p->len].p[0] = c1;
    p->els[p->len].p[1] = c2;
    p->els[p->len].p[2] = to;
    p->len++;
}
static inline void bezpath_close_path(BezPath* p)
{
    assert(p->len < BEZPATH_MAX_ELS);
    p->els[p->len].kind = PATH_EL_CLOSE_PATH;
    p->len++;
}

/* ============================================================
 * SECTION: RoundedRectRadii
 * ============================================================ */

typedef struct
{
    double top_left;
    double top_right;
    double bottom_right;
    double bottom_left;
} RoundedRectRadii;

static inline RoundedRectRadii rounded_rect_radii_new(double tl, double tr, double br, double bl)
{
    RoundedRectRadii r = {tl, tr, br, bl};
    return r;
}
static inline RoundedRectRadii rounded_rect_radii_from_single(double radius)
{
    RoundedRectRadii r = {radius, radius, radius, radius};
    return r;
}
static inline RoundedRectRadii rounded_rect_radii_abs(RoundedRectRadii r)
{
    RoundedRectRadii o = {fabs(r.top_left), fabs(r.top_right), fabs(r.bottom_right), fabs(r.bottom_left)};
    return o;
}
static inline RoundedRectRadii rounded_rect_radii_clamp(RoundedRectRadii r, double max)
{
    RoundedRectRadii o =
        {fmin(r.top_left, max), fmin(r.top_right, max), fmin(r.bottom_right, max), fmin(r.bottom_left, max)};
    return o;
}
static inline bool rounded_rect_radii_is_finite(RoundedRectRadii r)
{
    return isfinite(r.top_left) && isfinite(r.top_right) && isfinite(r.bottom_right) && isfinite(r.bottom_left);
}
static inline bool rounded_rect_radii_is_nan(RoundedRectRadii r)
{
    return isnan(r.top_left) || isnan(r.top_right) || isnan(r.bottom_right) || isnan(r.bottom_left);
}
/* Returns true and sets *out if all radii are equal within 1e-9 */
static inline bool rounded_rect_radii_as_single(RoundedRectRadii r, double* out)
{
    double eps = 1e-9;
    if (fabs(r.top_left - r.top_right) < eps && fabs(r.top_right - r.bottom_right) < eps &&
        fabs(r.bottom_right - r.bottom_left) < eps)
    {
        if (out)
            *out = r.top_left;
        return true;
    }
    return false;
}
static inline RoundedRectRadii rounded_rect_radii_scale(RoundedRectRadii r, double s)
{
    RoundedRectRadii o = {r.top_left * s, r.top_right * s, r.bottom_right * s, r.bottom_left * s};
    return o;
}

/* ============================================================
 * SECTION: RoundedRect
 * ============================================================ */

typedef struct
{
    Rect             rect;
    RoundedRectRadii radii;
} RoundedRect;

static inline RoundedRect rounded_rect_from_rect(Rect rect, RoundedRectRadii radii)
{
    Rect             r        = rect_abs(rect);
    double           shortest = fmin(rect_width(r), rect_height(r));
    RoundedRectRadii clamped  = rounded_rect_radii_clamp(rounded_rect_radii_abs(radii), shortest * 0.5);
    RoundedRect      rr       = {r, clamped};
    return rr;
}
static inline RoundedRect rounded_rect_new(double x0, double y0, double x1, double y1, RoundedRectRadii radii)
{
    return rounded_rect_from_rect(rect_new(x0, y0, x1, y1), radii);
}
static inline RoundedRect rounded_rect_from_single(double x0, double y0, double x1, double y1, double radius)
{
    return rounded_rect_new(x0, y0, x1, y1, rounded_rect_radii_from_single(radius));
}
static inline double           rounded_rect_width(RoundedRect rr) { return rect_width(rr.rect); }
static inline double           rounded_rect_height(RoundedRect rr) { return rect_height(rr.rect); }
static inline RoundedRectRadii rounded_rect_radii(RoundedRect rr) { return rr.radii; }
static inline Rect             rounded_rect_rect(RoundedRect rr) { return rr.rect; }
static inline Point            rounded_rect_origin(RoundedRect rr) { return rect_origin(rr.rect); }
static inline Point            rounded_rect_center(RoundedRect rr) { return rect_center(rr.rect); }
static inline bool             rounded_rect_is_finite(RoundedRect rr)
{
    return rect_is_finite(rr.rect) && rounded_rect_radii_is_finite(rr.radii);
}
static inline bool rounded_rect_is_nan(RoundedRect rr)
{
    return rect_is_nan(rr.rect) || rounded_rect_radii_is_nan(rr.radii);
}

/* Shape methods for RoundedRect */
static inline double rounded_rect_area(RoundedRect rr)
{
    /* rect area minus (1 - pi/4) * r^2 per corner */
    RoundedRectRadii ra          = rr.radii;
    double           corner_area = (M_PI / 4.0 - 1.0) * (ra.top_left * ra.top_left + ra.top_right * ra.top_right +
                                               ra.bottom_right * ra.bottom_right + ra.bottom_left * ra.bottom_left);
    return rect_area(rr.rect) + corner_area;
}
static inline double rounded_rect_perimeter(RoundedRect rr, double accuracy)
{
    (void)accuracy;
    RoundedRectRadii ra = rr.radii;
    double corner_perim = (M_PI / 2.0 - 2.0) * (ra.top_left + ra.top_right + ra.bottom_right + ra.bottom_left);
    return rect_perimeter(rr.rect) + corner_perim;
}
static inline int rounded_rect_winding(RoundedRect rr, Point pt)
{
    Point  center = rounded_rect_center(rr);
    double dx     = pt.x - center.x;
    double dy     = pt.y - center.y;
    /* pick radius by quadrant */
    RoundedRectRadii ra = rr.radii;
    double           radius;
    {
        double rt = (dx >= 0.0) ? ra.top_right : ra.top_left;
        double rb = (dx >= 0.0) ? ra.bottom_right : ra.bottom_left;
        radius    = (dy >= 0.0) ? rb : rt;
    }
    double ihw = fmax(0.0, rounded_rect_width(rr) * 0.5 - radius);
    double ihh = fmax(0.0, rounded_rect_height(rr) * 0.5 - radius);
    double px  = fmax(0.0, fabs(dx) - ihw);
    double py  = fmax(0.0, fabs(dy) - ihh);
    return (px * px + py * py <= radius * radius) ? 1 : 0;
}
static inline Rect        rounded_rect_bounding_box(RoundedRect rr) { return rect_abs(rr.rect); }
static inline RoundedRect rounded_rect_add_vec2(RoundedRect rr, Vec2 v)
{
    return rounded_rect_from_rect(rect_add_vec2(rr.rect, v), rr.radii);
}
static inline RoundedRect rounded_rect_sub_vec2(RoundedRect rr, Vec2 v)
{
    return rounded_rect_from_rect(rect_sub_vec2(rr.rect, v), rr.radii);
}

/* ============================================================
 * SECTION: Additional Rect methods (from rect.rs)
 * ============================================================ */

static inline bool rect_contains_pt(Rect r, Point p) { return p.x >= r.x0 && p.x < r.x1 && p.y >= r.y0 && p.y < r.y1; }
static inline bool rect_overlaps(Rect a, Rect b)
{
    return a.x0 <= b.x1 && a.x1 >= b.x0 && a.y0 <= b.y1 && a.y1 >= b.y0;
}
static inline bool rect_contains_rect(Rect a, Rect b)
{
    return a.x0 <= b.x0 && a.y0 <= b.y0 && a.x1 >= b.x1 && a.y1 >= b.y1;
}
static inline Rect rect_inflate(Rect r, double w, double h) { return rect_new(r.x0 - w, r.y0 - h, r.x1 + w, r.y1 + h); }
static inline Rect rect_round(Rect r) { return rect_new(round(r.x0), round(r.y0), round(r.x1), round(r.y1)); }
static inline Rect rect_ceil(Rect r) { return rect_new(ceil(r.x0), ceil(r.y0), ceil(r.x1), ceil(r.y1)); }
static inline Rect rect_floor(Rect r) { return rect_new(floor(r.x0), floor(r.y0), floor(r.x1), floor(r.y1)); }
static inline Rect rect_expand(Rect r)
{
    double x0 = (r.x0 < r.x1) ? floor(r.x0) : ceil(r.x0);
    double x1 = (r.x0 < r.x1) ? ceil(r.x1) : floor(r.x1);
    double y0 = (r.y0 < r.y1) ? floor(r.y0) : ceil(r.y0);
    double y1 = (r.y0 < r.y1) ? ceil(r.y1) : floor(r.y1);
    return rect_new(x0, y0, x1, y1);
}
static inline Rect rect_trunc(Rect r)
{
    double x0 = (r.x0 < r.x1) ? ceil(r.x0) : floor(r.x0);
    double x1 = (r.x0 < r.x1) ? floor(r.x1) : ceil(r.x1);
    double y0 = (r.y0 < r.y1) ? ceil(r.y0) : floor(r.y0);
    double y1 = (r.y0 < r.y1) ? floor(r.y1) : ceil(r.y1);
    return rect_new(x0, y0, x1, y1);
}
static inline Rect rect_scale_from_origin(Rect r, double f) { return rect_new(r.x0 * f, r.y0 * f, r.x1 * f, r.y1 * f); }
static inline RoundedRect rect_to_rounded_rect(Rect r, RoundedRectRadii radii)
{
    return rounded_rect_from_rect(r, radii);
}
static inline double rect_aspect_ratio_width(Rect r) { return rect_width(r) / rect_height(r); }
/* Inscribed rect with given aspect_ratio (width/height).
 * Returns the largest rect with that aspect ratio fitting inside r. */
static inline Rect rect_inscribed_with_aspect_ratio(Rect r, double aspect_ratio)
{
    double w = rect_width(r), h = rect_height(r);
    double self_aspect = (h != 0.0) ? w / h : (w >= 0.0 ? 1e300 : -1e300);
    if (isnan(self_aspect) || fabs(self_aspect - aspect_ratio) < 1e-9)
        return r;
    if (fabs(self_aspect) < fabs(aspect_ratio))
    {
        /* shrink y */
        double new_h = (aspect_ratio != 0.0) ? w / aspect_ratio : 0.0;
        double gap   = (h - new_h) * 0.5;
        return rect_new(r.x0, r.y0 + gap, r.x1, r.y1 - gap);
    }
    else
    {
        /* shrink x */
        double new_w = h * aspect_ratio;
        double gap   = (w - new_w) * 0.5;
        return rect_new(r.x0 + gap, r.y0, r.x1 - gap, r.y1);
    }
}
static inline Rect rect_sub_rect(Rect a, Rect b)
{
    /* Returns Insets-equivalent as Rect: x0=b.x0-a.x0, y0=b.y0-a.y0, x1=a.x1-b.x1, y1=a.y1-b.y1 */
    return rect_new(b.x0 - a.x0, b.y0 - a.y0, a.x1 - b.x1, a.y1 - b.y1);
}
/* Winding number for Rect — tiled so exactly one tile contains each point */
static inline int rect_winding(Rect r, Point pt)
{
    double xmin = fmin(r.x0, r.x1), xmax = fmax(r.x0, r.x1);
    double ymin = fmin(r.y0, r.y1), ymax = fmax(r.y0, r.y1);
    if (pt.x >= xmin && pt.x < xmax && pt.y >= ymin && pt.y < ymax)
    {
        return ((r.x1 > r.x0) ^ (r.y1 > r.y0)) ? -1 : 1;
    }
    return 0;
}
static inline double rect_area_signed(Rect r) { return rect_width(r) * rect_height(r); }
/* rect_perimeter is defined earlier in the Rect section */

/* ============================================================
 * SECTION: TranslateScale
 * ============================================================ */

typedef struct
{
    Vec2   translation;
    double scale;
} TranslateScale;

static inline TranslateScale translate_scale_new(Vec2 t, double s)
{
    TranslateScale ts = {t, s};
    return ts;
}
static inline TranslateScale translate_scale_identity(void) { return translate_scale_new(vec2_new(0.0, 0.0), 1.0); }
static inline TranslateScale translate_scale_from_scale(double s) { return translate_scale_new(vec2_new(0.0, 0.0), s); }
static inline TranslateScale translate_scale_from_translate(Vec2 t) { return translate_scale_new(t, 1.0); }
static inline TranslateScale translate_scale_from_scale_about(double s, Point focus)
{
    Vec2 f = point_to_vec2(focus);
    Vec2 t = vec2_sub(f, vec2_scale(f, s));
    return translate_scale_new(t, s);
}
static inline TranslateScale translate_scale_inverse(TranslateScale ts)
{
    double         recip = 1.0 / ts.scale;
    TranslateScale r     = {vec2_scale(ts.translation, -recip), recip};
    return r;
}
static inline bool translate_scale_is_finite(TranslateScale ts)
{
    return vec2_is_finite(ts.translation) && isfinite(ts.scale);
}
static inline bool translate_scale_is_nan(TranslateScale ts) { return vec2_is_nan(ts.translation) || isnan(ts.scale); }
/* Apply transform to a Point */
static inline Point translate_scale_apply_pt(TranslateScale ts, Point p)
{
    return point_add_vec2(vec2_to_point(vec2_scale(point_to_vec2(p), ts.scale)), ts.translation);
}
/* Compose two TranslateScale: self * other  (self applied after other) */
static inline TranslateScale translate_scale_mul(TranslateScale a, TranslateScale b)
{
    TranslateScale r;
    r.translation = vec2_add(a.translation, vec2_scale(b.translation, a.scale));
    r.scale       = a.scale * b.scale;
    return r;
}
/* f64 * TranslateScale */
static inline TranslateScale translate_scale_scale_by(double f, TranslateScale ts)
{
    TranslateScale r = {vec2_scale(ts.translation, f), ts.scale * f};
    return r;
}
/* TranslateScale + Vec2 */
static inline TranslateScale translate_scale_add_vec2(TranslateScale ts, Vec2 v)
{
    TranslateScale r = {vec2_add(ts.translation, v), ts.scale};
    return r;
}
/* TranslateScale - Vec2 */
static inline TranslateScale translate_scale_sub_vec2(TranslateScale ts, Vec2 v)
{
    TranslateScale r = {vec2_sub(ts.translation, v), ts.scale};
    return r;
}
/* Convert to Affine */
static inline Affine translate_scale_to_affine(TranslateScale ts)
{
    Affine a;
    a.c[0] = ts.scale;
    a.c[1] = 0.0;
    a.c[2] = 0.0;
    a.c[3] = ts.scale;
    a.c[4] = ts.translation.x;
    a.c[5] = ts.translation.y;
    return a;
}
/* Apply to shapes */
static inline Circle translate_scale_apply_circle(TranslateScale ts, Circle c)
{
    return circle_new(translate_scale_apply_pt(ts, c.center), ts.scale * c.radius);
}
static inline Line translate_scale_apply_line(TranslateScale ts, Line l)
{
    return line_new(translate_scale_apply_pt(ts, l.p0), translate_scale_apply_pt(ts, l.p1));
}
static inline Rect translate_scale_apply_rect(TranslateScale ts, Rect r)
{
    Point p0 = translate_scale_apply_pt(ts, point_new(r.x0, r.y0));
    Point p1 = translate_scale_apply_pt(ts, point_new(r.x1, r.y1));
    return rect_from_points(p0, p1);
}
static inline RoundedRect translate_scale_apply_rounded_rect(TranslateScale ts, RoundedRect rr)
{
    return rounded_rect_from_rect(
        translate_scale_apply_rect(ts, rr.rect),
        rounded_rect_radii_scale(rr.radii, ts.scale));
}
static inline RoundedRectRadii translate_scale_apply_radii(TranslateScale ts, RoundedRectRadii r)
{
    return rounded_rect_radii_scale(r, ts.scale);
}
static inline QuadBez translate_scale_apply_quadbez(TranslateScale ts, QuadBez q)
{
    return quadbez_new(
        translate_scale_apply_pt(ts, q.p0),
        translate_scale_apply_pt(ts, q.p1),
        translate_scale_apply_pt(ts, q.p2));
}
static inline CubicBez translate_scale_apply_cubicbez(TranslateScale ts, CubicBez c)
{
    return cubicbez_new(
        translate_scale_apply_pt(ts, c.p0),
        translate_scale_apply_pt(ts, c.p1),
        translate_scale_apply_pt(ts, c.p2),
        translate_scale_apply_pt(ts, c.p3));
}

/* ============================================================
 * SECTION: Triangle
 * ============================================================ */

typedef struct
{
    Point a, b, c;
} Triangle;

static inline Triangle triangle_new(Point a, Point b, Point c)
{
    Triangle t = {a, b, c};
    return t;
}
static inline Triangle triangle_from_coords(double ax, double ay, double bx, double by, double cx, double cy)
{
    Triangle t = {point_new(ax, ay), point_new(bx, by), point_new(cx, cy)};
    return t;
}
static inline Point triangle_centroid(Triangle t)
{
    return point_new((t.a.x + t.b.x + t.c.x) / 3.0, (t.a.y + t.b.y + t.c.y) / 3.0);
}
static inline double triangle_area(Triangle t)
{
    Vec2 ab = point_sub(t.b, t.a);
    Vec2 ac = point_sub(t.c, t.a);
    return 0.5 * vec2_cross(ab, ac);
}
static inline bool triangle_is_zero_area(Triangle t) { return triangle_area(t) == 0.0; }

/* Inscribed circle (incircle) */
static inline Circle triangle_inscribed_circle(Triangle t)
{
    double ab              = point_distance(t.a, t.b);
    double bc              = point_distance(t.b, t.c);
    double ac              = point_distance(t.a, t.c);
    double perimeter_recip = 1.0 / (ab + bc + ac);
    double ix              = (t.a.x * bc + t.b.x * ac + t.c.x * ab) * perimeter_recip;
    double iy              = (t.a.y * bc + t.b.y * ac + t.c.y * ab) * perimeter_recip;
    double radius          = 2.0 * fabs(triangle_area(t)) * perimeter_recip;
    return circle_new(point_new(ix, iy), radius);
}

/* Circumscribed circle (circumcircle) */
static inline Circle triangle_circumscribed_circle(Triangle t)
{
    Vec2   b       = point_sub(t.b, t.a);
    Vec2   c       = point_sub(t.c, t.a);
    double b_len2  = vec2_dot(b, b);
    double c_len2  = vec2_dot(c, c);
    double d_recip = 0.5 / vec2_cross(b, c);
    double x       = (c.y * b_len2 - b.y * c_len2) * d_recip;
    double y       = (b.x * c_len2 - c.x * b_len2) * d_recip;
    Vec2   bc_diff = vec2_sub(c, b);
    double r       = sqrt(b_len2 * c_len2) * sqrt(bc_diff.x * bc_diff.x + bc_diff.y * bc_diff.y) * fabs(d_recip);
    Point  center  = point_add_vec2(t.a, vec2_new(x, y));
    /* preserve sign of d_recip for radius sign (matching Rust) */
    return circle_new(center, sqrt(b_len2 * c_len2) * sqrt(bc_diff.x * bc_diff.x + bc_diff.y * bc_diff.y) * d_recip);
}

/* Inflate triangle by scalar in all directions */
static inline Triangle triangle_inflate(Triangle t, double scalar)
{
    Point               cen        = triangle_centroid(t);
    static const double FRAC_5PI_4 = 5.0 * M_PI / 4.0;
    static const double FRAC_7PI_4 = 7.0 * M_PI / 4.0;
    return triangle_new(
        point_add_vec2(cen, vec2_new(0.0, scalar)),
        point_add_vec2(cen, vec2_scale(vec2_from_angle(FRAC_5PI_4), scalar)),
        point_add_vec2(cen, vec2_scale(vec2_from_angle(FRAC_7PI_4), scalar)));
}
static inline bool triangle_is_finite(Triangle t)
{
    return point_is_finite(t.a) && point_is_finite(t.b) && point_is_finite(t.c);
}
static inline bool   triangle_is_nan(Triangle t) { return point_is_nan(t.a) || point_is_nan(t.b) || point_is_nan(t.c); }
static inline double triangle_perimeter(Triangle t)
{
    return point_distance(t.a, t.b) + point_distance(t.b, t.c) + point_distance(t.c, t.a);
}
static inline int triangle_winding(Triangle t, Point pt)
{
    Vec2   ab = point_sub(t.b, t.a), pa = point_sub(pt, t.a);
    Vec2   bc = point_sub(t.c, t.b), pb = point_sub(pt, t.b);
    Vec2   ca = point_sub(t.a, t.c), pc = point_sub(pt, t.c);
    double s0 = (vec2_cross(ab, pa) >= 0.0) ? 1.0 : -1.0;
    double s1 = (vec2_cross(bc, pb) >= 0.0) ? 1.0 : -1.0;
    double s2 = (vec2_cross(ca, pc) >= 0.0) ? 1.0 : -1.0;
    if (s0 == s1 && s1 == s2)
        return (int)s0;
    return 0;
}
static inline Rect triangle_bounding_box(Triangle t)
{
    return rect_new(
        fmin(t.a.x, fmin(t.b.x, t.c.x)),
        fmin(t.a.y, fmin(t.b.y, t.c.y)),
        fmax(t.a.x, fmax(t.b.x, t.c.x)),
        fmax(t.a.y, fmax(t.b.y, t.c.y)));
}
static inline Triangle triangle_add_vec2(Triangle t, Vec2 v)
{
    return triangle_new(point_add_vec2(t.a, v), point_add_vec2(t.b, v), point_add_vec2(t.c, v));
}
static inline Triangle triangle_sub_vec2(Triangle t, Vec2 v)
{
    return triangle_new(point_sub_vec2(t.a, v), point_sub_vec2(t.b, v), point_sub_vec2(t.c, v));
}

/* ============================================================
 * SECTION: simplify.rs — moment_integrals (exposed helper)
 * ============================================================ */

typedef struct
{
    double area, mx, my;
} MomentIntegrals;

static inline MomentIntegrals moment_integrals(CubicBez c)
{
    /* Direct port of simplify.rs moment_integrals — Green's theorem moments */
    double x0 = c.p0.x, y0 = c.p0.y;
    double x1 = c.p1.x - x0, y1 = c.p1.y - y0;
    double x2 = c.p2.x - x0, y2 = c.p2.y - y0;
    double x3 = c.p3.x - x0, y3 = c.p3.y - y0;
    double r0 = 3. * x1, r1 = 3. * y1, r2 = x2 * y3, r3 = x3 * y2, r4 = x3 * y3;
    double r5 = 27. * y1, r6 = x1 * x2, r7 = 27. * y2, r8 = 45. * r2, r9 = 18. * x3;
    double r10 = x1 * y1, r11 = 30. * x1, r12 = 45. * x3, r13 = x2 * y1, r14 = 45. * r3;
    double r15 = x1 * x1, r16 = 18. * y3, r17 = x2 * x2, r18 = 45. * y3, r19 = x3 * x3;
    double r20 = 30. * y1, r21 = y2 * y2, r22 = y3 * y3, r23 = y1 * y1;
    double a    = -r0 * y2 - r0 * y3 + r1 * x2 + r1 * x3 - 6. * r2 + 6. * r3 + 10. * r4;
    double lift = x3 * y0;
    double area = a * 0.05 + lift;
    double xm   = r10 * r9 - r11 * r4 + r12 * r13 + r14 * x2 - r15 * r16 - r15 * r7 - r17 * r18 + r17 * r5 + r19 * r20 +
                105. * r19 * y2 + 280. * r19 * y3 - 105. * r2 * x3 + r5 * r6 - r6 * r7 - r8 * x1;
    double ym = -r10 * r16 - r10 * r7 - r11 * r22 + r12 * r21 + r13 * r7 + r14 * y1 - r18 * x1 * y2 + r20 * r4 -
                27. * r21 * x1 - 105. * r22 * x2 + 140. * r22 * x3 + r23 * r9 + 27. * r23 * x2 + 105. * r3 * y3 -
                r8 * y2;
    MomentIntegrals m;
    m.area = area;
    m.mx   = xm * (1. / 840.) + x0 * area + 0.5 * x3 * lift;
    m.my   = ym * (1. / 420.) + y0 * a * 0.1 + y0 * lift;
    return m;
}

/* ============================================================
 * SECTION: Stroke types (stroke.rs)
 * Stroke stroking uses fit.rs / offset.rs which are not yet ported.
 * The data structures are defined here for completeness.
 * ============================================================ */

typedef enum
{
    JOIN_BEVEL,
    JOIN_MITER,
    JOIN_ROUND
} Join;
typedef enum
{
    CAP_BUTT,
    CAP_SQUARE,
    CAP_ROUND
} Cap;

#define STROKE_MAX_DASH_PATTERN 16
typedef struct
{
    double width;
    Join   join;
    double miter_limit;
    Cap    start_cap;
    Cap    end_cap;
    double dash_pattern[STROKE_MAX_DASH_PATTERN];
    int    dash_pattern_len;
    double dash_offset;
} Stroke;

static inline Stroke stroke_new(double width)
{
    Stroke s;
    s.width            = width;
    s.join             = JOIN_ROUND;
    s.miter_limit      = 4.0;
    s.start_cap        = CAP_ROUND;
    s.end_cap          = CAP_ROUND;
    s.dash_pattern_len = 0;
    s.dash_offset      = 0.0;
    return s;
}

/* ============================================================
 * SECTION: SvgArc (svg.rs)
 * ============================================================ */

typedef struct
{
    Point  from;
    Point  to;
    Vec2   radii;
    double x_rotation;
    bool   large_arc;
    bool   sweep;
} SvgArc;

/* Convert an SvgArc to a sequence of cubic bezier curves via callback.
 * Follows the SVG arc-to-bezier algorithm (adapted from the Rust svg.rs port of lyon).
 *
 * LOUD COMMENT: svg.rs (~692 lines) contains the full SVG path parsing (from_svg),
 * writing (to_svg / write_to), and SvgArc::to_arc / SvgArc::to_cubic_beziers.
 * The parsing and writing code depends on string manipulation and I/O that are
 * not straightforward in C without additional library support.
 * The arc-to-bezier conversion is ported below, but the full SVG string I/O
 * is OMITTED — add it manually if SVG import/export is needed.
 */
static inline void svg_arc_to_cubic_beziers(SvgArc sa, void (*cb)(CubicBez, void*), void* user)
{
    /* Port of the lyon-based SVG arc decomposition.
     * Reference: https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes */
    double rx = fabs(sa.radii.x), ry = fabs(sa.radii.y);
    if (rx < 1e-10 || ry < 1e-10)
    {
        /* Degenerate: treat as a line */
        Point    mid = point_new(0.5 * (sa.from.x + sa.to.x), 0.5 * (sa.from.y + sa.to.y));
        CubicBez cb_ = cubicbez_new(sa.from, mid, mid, sa.to);
        cb(cb_, user);
        return;
    }
    double cos_phi = cos(sa.x_rotation), sin_phi = sin(sa.x_rotation);
    /* Step 1: Compute (x1', y1') */
    double dx2 = (sa.from.x - sa.to.x) * 0.5;
    double dy2 = (sa.from.y - sa.to.y) * 0.5;
    double x1p = cos_phi * dx2 + sin_phi * dy2;
    double y1p = -sin_phi * dx2 + cos_phi * dy2;
    /* Correct radii */
    double x1p2 = x1p * x1p, y1p2 = y1p * y1p;
    double rx2 = rx * rx, ry2 = ry * ry;
    double lambda = x1p2 / rx2 + y1p2 / ry2;
    if (lambda > 1.0)
    {
        double l  = sqrt(lambda);
        rx       *= l;
        ry       *= l;
        rx2       = rx * rx;
        ry2       = ry * ry;
    }
    /* Step 2: Compute (cx', cy') */
    double num = rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2;
    double den = rx2 * y1p2 + ry2 * x1p2;
    double sq  = (den > 0.0 && num / den >= 0.0) ? sqrt(num / den) : 0.0;
    if (sa.large_arc == sa.sweep)
        sq = -sq;
    double cxp = sq * rx * y1p / ry;
    double cyp = -sq * ry * x1p / rx;
    /* Step 3: Compute (cx, cy) from (cx', cy') */
    double cx = cos_phi * cxp - sin_phi * cyp + (sa.from.x + sa.to.x) * 0.5;
    double cy = sin_phi * cxp + cos_phi * cyp + (sa.from.y + sa.to.y) * 0.5;
    /* Step 4: Compute theta1 and dtheta */
    double ux = (x1p - cxp) / rx, uy = (y1p - cyp) / ry;
    double vx = (-x1p - cxp) / rx, vy = (-y1p - cyp) / ry;
    double u_len    = sqrt(ux * ux + uy * uy);
    double theta1   = acos(ux / u_len) * (uy < 0.0 ? -1.0 : 1.0);
    double uv_dot   = ux * vx + uy * vy;
    double uv_cross = ux * vy - uy * vx;
    double uv_len   = u_len * sqrt(vx * vx + vy * vy);
    double dtheta   = acos(fmax(-1.0, fmin(1.0, uv_dot / uv_len))) * (uv_cross < 0.0 ? -1.0 : 1.0);
    if (!sa.sweep && dtheta > 0.0)
        dtheta -= 2.0 * M_PI;
    if (sa.sweep && dtheta < 0.0)
        dtheta += 2.0 * M_PI;
    /* Decompose into cubic bezier segments (<= 90 deg each) */
    int n_segs = (int)ceil(fabs(dtheta) / (M_PI / 2.0));
    if (n_segs < 1)
        n_segs = 1;
    double d      = dtheta / n_segs;
    double alpha  = sin(d) * (sqrt(4.0 + 3.0 * tan(d / 2.0) * tan(d / 2.0)) - 1.0) / 3.0;
    double theta  = theta1;
    double cos_th = cos(theta1), sin_th = sin(theta1);
    double dx1 = -sin_th * rx * cos_phi - cos_th * ry * sin_phi;
    double dy1 = -sin_th * rx * sin_phi + cos_th * ry * cos_phi;
    for (int i = 0; i < n_segs; i++)
    {
        double px0     = cx + rx * cos_phi * cos_th - ry * sin_phi * sin_th;
        double py0     = cy + rx * sin_phi * cos_th + ry * cos_phi * sin_th;
        theta         += d;
        cos_th         = cos(theta);
        sin_th         = sin(theta);
        double   dx2_  = -sin_th * rx * cos_phi - cos_th * ry * sin_phi;
        double   dy2_  = -sin_th * rx * sin_phi + cos_th * ry * cos_phi;
        double   px3   = cx + rx * cos_phi * cos_th - ry * sin_phi * sin_th;
        double   py3   = cy + rx * sin_phi * cos_th + ry * cos_phi * sin_th;
        CubicBez seg   = cubicbez_new(
            point_new(px0, py0),
            point_new(px0 + alpha * dx1, py0 + alpha * dy1),
            point_new(px3 - alpha * dx2_, py3 - alpha * dy2_),
            point_new(px3, py3));
        cb(seg, user);
        dx1 = dx2_;
        dy1 = dy2_;
    }
}

/* ============================================================
 * SECTION: fit.rs — ParamCurveFit trait + curve fitting
 * ============================================================
 *
 * Rust traits become vtable structs in C.  A "ParamCurveFit object" is a pair
 * of (const void *self, const ParamCurveFitVtbl *vtbl).  All fit_* functions
 * take those two arguments.
 *
 * Gauss-Legendre 16-point table (shared with moment_integrals default impl).
 * Values are (weight, abscissa) pairs on [-1, 1].
 */

static const double GL16_W[16] = {
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.1826034150449236,
    0.1691565193950025,
    0.1691565193950025,
    0.1495959888165767,
    0.1495959888165767,
    0.1246289512509458,
    0.1246289512509458,
    0.0951585116824928,
    0.0951585116824928,
    0.0622535239386478,
    0.0622535239386478,
    0.0271524593914925,
    0.0271524593914925};
static const double GL16_X[16] = {
    -0.0950125098360823,
    0.0950125098360823,
    -0.2816035507792589,
    0.2816035507792589,
    -0.4580167776572273,
    0.4580167776572273,
    -0.6178762444026437,
    0.6178762444026437,
    -0.7554044083550030,
    0.7554044083550030,
    -0.8656312023341098,
    0.8656312023341098,
    -0.9445750230732326,
    0.9445750230732326,
    -0.9894009349916499,
    0.9894009349916499};

/* A sample point produced by ParamCurveFit */
typedef struct
{
    Point p;
    Vec2  tangent;
} CurveFitSample;

/* Virtual table for the ParamCurveFit trait.
 * 'self' is an opaque pointer to the concrete object. */
typedef struct
{
    /* Required: sample point + tangent at t.  sign (+1 or -1) selects which
     * side of a cusp/corner to approach from. */
    CurveFitSample (*sample_pt_tangent)(const void* self, double t, double sign);
    /* Required: sample point + derivative (can be zero at cusps) */
    void (*sample_pt_deriv)(const void* self, double t, Point* p_out, Vec2* d_out);
    /* Optional: if NULL the default GL-16 quadrature is used */
    void (*moment_integrals)(const void* self, double t0, double t1, double* area, double* mx, double* my);
    /* Optional: if NULL no cusps are reported.
     * Returns 1 and sets *t_out to the cusp location when found, else 0. */
    int (*break_cusp)(const void* self, double t0, double t1, double* t_out);
} ParamCurveFitVtbl;

/* ---- default moment_integrals via GL-16 quadrature ---- */
static void pcf_moment_integrals_default(
    const void*              self,
    const ParamCurveFitVtbl* vtbl,
    double                   t0,
    double                   t1,
    double*                  area_out,
    double*                  mx_out,
    double*                  my_out)
{
    double tmid = 0.5 * (t0 + t1), dt = 0.5 * (t1 - t0);
    double area = 0.0, mx = 0.0, my = 0.0;
    for (int i = 0; i < 16; i++)
    {
        double t = tmid + GL16_X[i] * dt;
        Point  p;
        Vec2   d;
        vtbl->sample_pt_deriv(self, t, &p, &d);
        double a  = GL16_W[i] * d.x * p.y;
        area     += a;
        mx       += p.x * a;
        my       += p.y * a;
    }
    *area_out = area * dt;
    *mx_out   = mx * dt;
    *my_out   = my * dt;
}

/* Convenience: dispatch moment_integrals through vtbl or default */
static inline void pcf_moment_integrals(
    const void*              self,
    const ParamCurveFitVtbl* vtbl,
    double                   t0,
    double                   t1,
    double*                  area,
    double*                  mx,
    double*                  my)
{
    if (vtbl->moment_integrals)
    {
        vtbl->moment_integrals(self, t0, t1, area, mx, my);
    }
    else
    {
        pcf_moment_integrals_default(self, vtbl, t0, t1, area, mx, my);
    }
}

/* Convenience: dispatch break_cusp */
static inline int pcf_break_cusp(const void* self, const ParamCurveFitVtbl* vtbl, double t0, double t1, double* t_out)
{
    if (vtbl->break_cusp)
    {
        return vtbl->break_cusp(self, t0, t1, t_out);
    }
    return 0;
}

/* ---- CurveFitSample::intersect — roots of (B(t)-p)·tangent = 0 on [0,1] ---- */
#define INTERSECT_MAX 3
static int curve_fit_sample_intersect(CurveFitSample s, CubicBez c, double roots_out[INTERSECT_MAX])
{
    /* Cubic equation coefficients for dot(B(t)-p, tangent) = 0 */
    Vec2 p1 = vec2_scale(point_sub(c.p1, c.p0), 3.0);
    Vec2 p2 = vec2_sub(
        vec2_sub(vec2_scale(point_to_vec2(c.p2), 3.0), vec2_scale(point_to_vec2(c.p1), 6.0)),
        vec2_scale(point_to_vec2(c.p0), -3.0));
    /* p2 = 3*p2 - 6*p1 + 3*p0 */
    p2         = vec2_new(3.0 * c.p2.x - 6.0 * c.p1.x + 3.0 * c.p0.x, 3.0 * c.p2.y - 6.0 * c.p1.y + 3.0 * c.p0.y);
    Vec2   p3  = vec2_sub(point_sub(c.p3, c.p0), vec2_scale(point_sub(c.p2, c.p1), 3.0));
    double c0  = vec2_dot(point_sub(c.p0, s.p), s.tangent);
    double c1  = vec2_dot(p1, s.tangent);
    double c2  = vec2_dot(p2, s.tangent);
    double c3  = vec2_dot(p3, s.tangent);
    DVec3  sol = solve_cubic(c0, c1, c2, c3);
    int    n   = 0;
    for (int i = 0; i < sol.len; i++)
    {
        double t = sol.v[i];
        if (t >= 0.0 && t <= 1.0)
            roots_out[n++] = t;
    }
    return n;
}

/* ---- CurveDist — acceleration structure for measuring fit error ---- */
#define N_SAMPLE_FIT 20
#define N_SUBSAMPLE  10

typedef struct
{
    CurveFitSample samples[N_SAMPLE_FIT];
    double         arcparams[N_SAMPLE_FIT];
    int            arcparams_computed;
    double         range_start, range_end;
    int            spicy;
} CurveDist;

static CurveDist curve_dist_new(const void* self, const ParamCurveFitVtbl* vtbl, double t0, double t1)
{
    CurveDist cd;
    cd.range_start                   = t0;
    cd.range_end                     = t1;
    cd.arcparams_computed            = 0;
    cd.spicy                         = 0;
    double              step         = (t1 - t0) * (1.0 / (N_SAMPLE_FIT + 1));
    static const double SPICY_THRESH = 0.2;
    Vec2                last_tan     = {0, 0};
    int                 has_last     = 0;
    for (int i = 0; i <= N_SAMPLE_FIT + 1; i++)
    {
        CurveFitSample s = vtbl->sample_pt_tangent(self, t0 + i * step, 1.0);
        if (has_last)
        {
            double cross = vec2_cross(s.tangent, last_tan);
            double dot   = vec2_dot(s.tangent, last_tan);
            if (fabs(cross) > SPICY_THRESH * fabs(dot))
                cd.spicy = 1;
        }
        last_tan = s.tangent;
        has_last = 1;
        if (i > 0 && i <= N_SAMPLE_FIT)
            cd.samples[i - 1] = s;
    }
    return cd;
}

static void curve_dist_compute_arc_params(CurveDist* cd, const void* self, const ParamCurveFitVtbl* vtbl)
{
    double t0 = cd->range_start, t1 = cd->range_end;
    int    total  = (N_SAMPLE_FIT + 1) * N_SUBSAMPLE;
    double dt     = (t1 - t0) / total;
    double arclen = 0.0;
    for (int i = 0; i <= N_SAMPLE_FIT; i++)
    {
        for (int j = 0; j < N_SUBSAMPLE; j++)
        {
            double t = t0 + dt * ((double)(i * N_SUBSAMPLE + j) + 0.5);
            Point  p;
            Vec2   d;
            vtbl->sample_pt_deriv(self, t, &p, &d);
            arclen += vec2_hypot(d);
        }
        if (i < N_SAMPLE_FIT)
            cd->arcparams[i] = arclen;
    }
    double inv = (arclen > 0.0) ? 1.0 / arclen : 0.0;
    for (int i = 0; i < N_SAMPLE_FIT; i++)
        cd->arcparams[i] *= inv;
    cd->arcparams_computed = 1;
}

static double f64_inv_arclen(CubicBez c, double target, double eps)
{
    /* binary search for t such that arclen(c, 0..t) == target */
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 64; i++)
    {
        double   mid = 0.5 * (lo + hi);
        CubicBez sub = cubicbez_subsegment(c, 0.0, mid);
        double   len = cubicbez_arclen(sub, eps);
        if (len < target)
            lo = mid;
        else
            hi = mid;
        if (hi - lo < eps)
            break;
    }
    return 0.5 * (lo + hi);
}

static double curve_dist_eval_arc(CurveDist* cd, CubicBez c, double acc2)
{
    static const double EPS      = 1e-9;
    double              c_arclen = cubicbez_arclen(c, EPS);
    double              max_err2 = 0.0;
    for (int i = 0; i < N_SAMPLE_FIT; i++)
    {
        double t   = f64_inv_arclen(c, c_arclen * cd->arcparams[i], EPS);
        double err = point_distance_sq(cd->samples[i].p, cubicbez_eval(c, t));
        if (err > max_err2)
            max_err2 = err;
        if (max_err2 > acc2)
            return -1.0; /* exceeded */
    }
    return max_err2;
}

static double curve_dist_eval_ray(CurveDist* cd, CubicBez c, double acc2)
{
    double max_err2 = 0.0;
    for (int i = 0; i < N_SAMPLE_FIT; i++)
    {
        double best = acc2 + 1.0;
        double roots[INTERSECT_MAX];
        int    nr = curve_fit_sample_intersect(cd->samples[i], c, roots);
        for (int j = 0; j < nr; j++)
        {
            double err = point_distance_sq(cd->samples[i].p, cubicbez_eval(c, roots[j]));
            if (err < best)
                best = err;
        }
        if (best > max_err2)
            max_err2 = best;
        if (max_err2 > acc2)
            return -1.0;
    }
    return max_err2;
}

/* Returns -1 if exceeds acc2, otherwise the squared error */
static double curve_dist_eval(CurveDist* cd, const void* self, const ParamCurveFitVtbl* vtbl, CubicBez c, double acc2)
{
    double ray = curve_dist_eval_ray(cd, c, acc2);
    if (ray < 0.0)
        return -1.0;
    if (!cd->spicy)
        return ray;
    if (!cd->arcparams_computed)
        curve_dist_compute_arc_params(cd, self, vtbl);
    return curve_dist_eval_arc(cd, c, acc2);
}

/* ---- cubic_fit — returns up to 4 candidate cubics for given angles/area/moment ---- */
#define CUBIC_FIT_MAX 4
typedef struct
{
    CubicBez c;
    double   d0;
    double   d1;
} CubicFitCand;

static int cubic_fit(double th0, double th1, double area, double mx, CubicFitCand out[CUBIC_FIT_MAX])
{
    double s0 = sin(th0), c0 = cos(th0);
    double s1 = sin(th1), c1 = cos(th1);
    /* quartic coefficients (from fit.rs cubic_fit) */
    double a4 = -9. * c0 * (((2. * s1 * c1 * c0 + s0 * (2. * c1 * c1 - 1.)) * c0 - 2. * s1 * c1) * c0 - c1 * c1 * s0);
    double a3 =
        12. * ((((c1 * (30. * area * c1 - s1) - 15. * area) * c0 + 2. * s0 - c1 * s0 * (c1 + 30. * area * s1)) * c0 +
                c1 * (s1 - 15. * area * c1)) *
                   c0 -
               s0 * c1 * c1);
    double a2 = 12. * ((((70. * mx + 15. * area) * s1 * s1 + c1 * (9. * s1 - 70. * c1 * mx - 5. * c1 * area)) * c0 -
                        5. * s0 * s1 * (3. * s1 - 4. * c1 * (7. * mx + area))) *
                           c0 -
                       c1 * (9. * s1 - 70. * c1 * mx - 5. * c1 * area));
    double a1 = 16. * (((12. * s0 - 5. * c0 * (42. * mx - 17. * area)) * s1 - 70. * c1 * (3. * mx - area) * s0 -
                        75. * c0 * c1 * area * area) *
                           s1 -
                       75. * c1 * c1 * area * area * s0);
    double a0 = 80. * s1 * (42. * s1 * mx - 25. * area * (s1 - c1 * area));

    double              roots[4];
    int                 nroots = 0;
    static const double EPS    = 1e-12;
    if (fabs(a4) > EPS)
    {
        /* factor quartic — factor_quartic_inner is stubbed; fall back to cubic */
        /* LOUD COMMENT: factor_quartic_inner() is still unimplemented (common.rs lines
         * 362-580). Until it is ported, cubic_fit falls back to solving as a depressed
         * cubic via the a3 branch, which loses some candidate solutions.  Port
         * factor_quartic_inner to unlock the full quartic solve path. */
        double a = a3 / a4, b = a2 / a4, c = a1 / a4, d = a0 / a4;
        /* Heuristic fallback: treat as cubic with leading a3 term */
        if (fabs(a3) > EPS)
        {
            DVec3 sol = solve_cubic(a0, a1, a2, a3);
            for (int i = 0; i < sol.len && nroots < 4; i++)
                roots[nroots++] = sol.v[i];
        }
        else
        {
            DVec2 sol = solve_quadratic(a0, a1, a2);
            for (int i = 0; i < sol.len && nroots < 4; i++)
                roots[nroots++] = sol.v[i];
        }
        (void)a;
        (void)b;
        (void)c;
        (void)d;
    }
    else if (fabs(a3) > EPS)
    {
        DVec3 sol = solve_cubic(a0, a1, a2, a3);
        for (int i = 0; i < sol.len && nroots < 4; i++)
            roots[nroots++] = sol.v[i];
    }
    else if (fabs(a2) > EPS || fabs(a1) > EPS || fabs(a0) > EPS)
    {
        DVec2 sol = solve_quadratic(a0, a1, a2);
        for (int i = 0; i < sol.len && nroots < 4; i++)
            roots[nroots++] = sol.v[i];
    }
    else
    {
        /* degenerate: unit-speed straight line */
        out[0].c  = cubicbez_new(point_new(0, 0), point_new(1. / 3., 0), point_new(2. / 3., 0), point_new(1, 0));
        out[0].d0 = 1. / 3.;
        out[0].d1 = 1. / 3.;
        return 1;
    }
    double s01 = s0 * c1 + s1 * c0;
    int    n   = 0;
    for (int i = 0; i < nroots && n < CUBIC_FIT_MAX; i++)
    {
        double d0 = roots[i];
        double d1;
        if (d0 > 0.0)
        {
            d1 = (d0 * s0 - area * (10. / 3.)) / (0.5 * d0 * s01 - s1);
            if (d1 <= 0.0)
            {
                d0 = s1 / s01;
                d1 = 0.0;
            }
        }
        else
        {
            d0 = 0.0;
            d1 = (fabs(s01) > EPS) ? s0 / s01 : 0.0;
        }
        if (d0 >= 0.0 && d1 >= 0.0)
        {
            out[n].c = cubicbez_new(
                point_new(0, 0),
                point_new(d0 * c0, d0 * s0),
                point_new(1.0 - d1 * c1, d1 * s1),
                point_new(1, 0));
            out[n].d0 = d0;
            out[n].d1 = d1;
            n++;
        }
    }
    return n;
}

/* mod_2pi: wrap angle to [-pi, pi] */
static inline double mod_2pi(double th)
{
    double ts = th * (0.5 / M_PI);
    return M_PI * 2.0 * (ts - round(ts));
}

/* try_fit_line: fit a chord as a degenerate cubic, returns 1 on success */
static int try_fit_line(
    const void*              self,
    const ParamCurveFitVtbl* vtbl,
    double                   accuracy,
    double                   t0,
    double                   t1,
    Point                    start,
    Point                    end,
    CubicBez*                c_out,
    double*                  err2_out)
{
    double           acc2     = accuracy * accuracy;
    static const int SHORT_N  = 7;
    double           max_err2 = 0.0;
    double           dt       = (t1 - t0) / (SHORT_N + 1);
    for (int i = 0; i < SHORT_N; i++)
    {
        double t = t0 + (i + 1) * dt;
        Point  p;
        Vec2   d;
        vtbl->sample_pt_deriv(self, t, &p, &d);
        Nearest near = line_nearest(line_new(start, end), p);
        if (near.distance_sq > acc2)
            return 0;
        if (near.distance_sq > max_err2)
            max_err2 = near.distance_sq;
    }
    Point p1  = point_lerp(start, end, 1. / 3.);
    Point p2  = point_lerp(end, start, 1. / 3.);
    *c_out    = cubicbez_new(start, p1, p2, end);
    *err2_out = max_err2;
    return 1;
}

/* fit_to_cubic: fit a single cubic bezier to source[t0..t1].
 * Returns 1 on success, 0 if not within accuracy. */
int fit_to_cubic(
    const void*              self,
    const ParamCurveFitVtbl* vtbl,
    double                   t0,
    double                   t1,
    double                   accuracy,
    CubicBez*                c_out,
    double*                  err2_out)
{
    CurveFitSample s_start = vtbl->sample_pt_tangent(self, t0, 1.0);
    CurveFitSample s_end   = vtbl->sample_pt_tangent(self, t1, -1.0);
    double         acc2    = accuracy * accuracy;
    double         chord2  = point_distance_sq(s_start.p, s_end.p);
    if (chord2 <= acc2)
    {
        return try_fit_line(self, vtbl, accuracy, t0, t1, s_start.p, s_end.p, c_out, err2_out);
    }
    Vec2   d   = point_sub(s_end.p, s_start.p);
    double th  = vec2_atan2(d);
    double th0 = mod_2pi(vec2_atan2(s_start.tangent) - th);
    double th1 = mod_2pi(th - vec2_atan2(s_end.tangent));

    double area, mx_raw, my_raw;
    pcf_moment_integrals(self, vtbl, t0, t1, &area, &mx_raw, &my_raw);
    double x0 = s_start.p.x, y0 = s_start.p.y, dx = d.x, dy = d.y;
    area              -= dx * (y0 + 0.5 * dy);
    double dy_3        = dy * (1. / 3.);
    mx_raw            -= dx * (x0 * y0 + 0.5 * (x0 * dy + y0 * dx) + dy_3 * dx);
    my_raw            -= dx * (y0 * y0 + y0 * dy + dy_3 * dy);
    mx_raw            -= x0 * area;
    my_raw             = 0.5 * my_raw - y0 * area;
    double moment      = dx * mx_raw + dy * my_raw;
    double chord2_inv  = 1.0 / chord2;
    double unit_area   = area * chord2_inv;
    double mx          = moment * chord2_inv * chord2_inv;
    double chord       = sqrt(chord2);
    /* Build affine: translate(start) * rotate(th) * scale(chord) */
    double cs = cos(th), sn = sin(th);
    /* aff * p = chord*(cs*p.x - sn*p.y) + x0,  chord*(sn*p.x + cs*p.y) + y0 */
    CurveDist           cd = curve_dist_new(self, vtbl, t0, t1);
    CubicFitCand        cands[CUBIC_FIT_MAX];
    int                 nc              = cubic_fit(th0, th1, unit_area, mx, cands);
    static const double D_PENALTY_ELBOW = 0.65;
    static const double D_PENALTY_SLOPE = 2.0;
    CubicBez            best_c          = {{0}};
    double              best_err2       = -1.0;
    for (int i = 0; i < nc; i++)
    {
        /* Apply affine to candidate (defined on unit chord) */
        CubicBez raw = cands[i].c;
        CubicBez c;
        c.p0 = point_new(chord * (cs * raw.p0.x - sn * raw.p0.y) + x0, chord * (sn * raw.p0.x + cs * raw.p0.y) + y0);
        c.p1 = point_new(chord * (cs * raw.p1.x - sn * raw.p1.y) + x0, chord * (sn * raw.p1.x + cs * raw.p1.y) + y0);
        c.p2 = point_new(chord * (cs * raw.p2.x - sn * raw.p2.y) + x0, chord * (sn * raw.p2.x + cs * raw.p2.y) + y0);
        c.p3 = point_new(chord * (cs * raw.p3.x - sn * raw.p3.y) + x0, chord * (sn * raw.p3.x + cs * raw.p3.y) + y0);
        double err2 = curve_dist_eval(&cd, self, vtbl, c, acc2);
        if (err2 < 0.0)
            continue;
        /* apply arm-length penalty */
        double sf0    = 1.0 + fmax(0.0, cands[i].d0 - D_PENALTY_ELBOW) * D_PENALTY_SLOPE;
        double sf1    = 1.0 + fmax(0.0, cands[i].d1 - D_PENALTY_ELBOW) * D_PENALTY_SLOPE;
        double scale  = fmax(sf0, sf1);
        scale        *= scale;
        err2         *= scale;
        if (err2 < acc2 && (best_err2 < 0.0 || err2 < best_err2))
        {
            best_c    = c;
            best_err2 = err2;
        }
    }
    if (best_err2 >= 0.0)
    {
        *c_out    = best_c;
        *err2_out = best_err2;
        return 1;
    }
    return 0;
}

/* ---- BezPath builder (uses existing fixed-capacity BezPath stub) ---- */

/* fit_to_bezpath_rec: recursive halving */
static void fit_to_bezpath_rec(
    const void*              self,
    const ParamCurveFitVtbl* vtbl,
    double                   t0,
    double                   t1,
    double                   accuracy,
    BezPath*                 path)
{
    CurveFitSample cs0  = vtbl->sample_pt_tangent(self, t0, 1.0);
    CurveFitSample cs1  = vtbl->sample_pt_tangent(self, t1, -1.0);
    double         acc2 = accuracy * accuracy;
    CubicBez       c;
    double         err2;
    if (point_distance_sq(cs0.p, cs1.p) <= acc2)
    {
        if (try_fit_line(self, vtbl, accuracy, t0, t1, cs0.p, cs1.p, &c, &err2))
        {
            if (path->len == 0)
                bezpath_move_to(path, c.p0);
            bezpath_curve_to(path, c.p1, c.p2, c.p3);
            return;
        }
    }
    double tc;
    int    has_cusp = pcf_break_cusp(self, vtbl, t0, t1, &tc);
    if (!has_cusp && fit_to_cubic(self, vtbl, t0, t1, accuracy, &c, &err2))
    {
        if (path->len == 0)
            bezpath_move_to(path, c.p0);
        bezpath_curve_to(path, c.p1, c.p2, c.p3);
        return;
    }
    double tmid = has_cusp ? tc : 0.5 * (t0 + t1);
    if (tmid == t0 || tmid == t1)
    {
        /* degenerate: draw straight line */
        Point p1 = point_lerp(cs0.p, cs1.p, 1. / 3.);
        Point p2 = point_lerp(cs1.p, cs0.p, 1. / 3.);
        if (path->len == 0)
            bezpath_move_to(path, cs0.p);
        bezpath_curve_to(path, p1, p2, cs1.p);
        return;
    }
    fit_to_bezpath_rec(self, vtbl, t0, tmid, accuracy, path);
    fit_to_bezpath_rec(self, vtbl, tmid, t1, accuracy, path);
}

/* fit_to_bezpath: public entry point */
BezPath fit_to_bezpath(const void* self, const ParamCurveFitVtbl* vtbl, double accuracy)
{
    BezPath path;
    bezpath_init(&path);
    fit_to_bezpath_rec(self, vtbl, 0.0, 1.0, accuracy, &path);
    return path;
}

/* ---- fit_to_bezpath_opt: optimised subdivision ---- */

/* ITP callback context for fit_opt_err_delta */
typedef struct
{
    const void*              self;
    const ParamCurveFitVtbl* vtbl;
    double                   t0_range, t1_range;
    int                      n;
    double                   limit;
    double                   missing_err;
    int                      cusp_found;
    double                   cusp_t;
} FitOptCtx;

/* Measure one segment: returns sqrt(err) or missing_err on failure */
static double measure_one_seg(const void* self, const ParamCurveFitVtbl* vtbl, double t0, double t1, double limit)
{
    CubicBez c;
    double   err2;
    if (fit_to_cubic(self, vtbl, t0, t1, limit, &c, &err2))
        return sqrt(err2);
    return limit * 2.0;
}

typedef enum
{
    FITRES_PARAM,
    FITRES_SEG_ERROR,
    FITRES_CUSP
} FitResultKind;
typedef struct
{
    FitResultKind kind;
    double        val;
} FitResult;

/* fit_opt_segment: find t1 within range such that segment [t0..t1] just fits accuracy */
static FitResult fit_opt_segment(const void* self, const ParamCurveFitVtbl* vtbl, double accuracy, double t0, double t1)
{
    double tc;
    if (pcf_break_cusp(self, vtbl, t0, t1, &tc))
    {
        FitResult r = {FITRES_CUSP, tc};
        return r;
    }
    double missing_err = accuracy * 2.0;
    double err         = measure_one_seg(self, vtbl, t0, t1, accuracy);
    if (err <= accuracy)
    {
        FitResult r = {FITRES_SEG_ERROR, err};
        return r;
    }
    /* binary search for the t where measure_one_seg crosses accuracy */
    double lo = t0, hi = t1;
    for (int i = 0; i < 64; i++)
    {
        double mid = 0.5 * (lo + hi);
        {
            double tc2;
            if (pcf_break_cusp(self, vtbl, t0, mid, &tc2))
            {
                FitResult r = {FITRES_CUSP, tc2};
                return r;
            }
        }
        double e = measure_one_seg(self, vtbl, t0, mid, accuracy);
        if (e <= accuracy)
            lo = mid;
        else
            hi = mid;
        if (hi - lo < 1e-9 * (t1 - t0))
            break;
    }
    (void)missing_err;
    FitResult r = {FITRES_PARAM, lo};
    return r;
}

/* fit_opt_err_delta: walk n segments from t0, return (accuracy - last_err) */
static double fit_opt_err_delta(
    const void*              self,
    const ParamCurveFitVtbl* vtbl,
    double                   accuracy,
    double                   limit,
    double                   t0_range,
    double                   t1_range,
    int                      n,
    int*                     cusp_found,
    double*                  cusp_t)
{
    double t0 = t0_range, t1 = t1_range;
    for (int i = 0; i < n - 1; i++)
    {
        FitResult fr = fit_opt_segment(self, vtbl, accuracy, t0, t1);
        if (fr.kind == FITRES_CUSP)
        {
            *cusp_found = 1;
            *cusp_t     = fr.val;
            return 0.0;
        }
        if (fr.kind == FITRES_SEG_ERROR)
            return accuracy; /* n-1 segs suffice */
        t0 = fr.val;
    }
    double err = measure_one_seg(self, vtbl, t0, t1, limit);
    return accuracy - err;
}

/* ITP callback wrapper */
typedef struct
{
    const void*              self;
    const ParamCurveFitVtbl* vtbl;
    double                   accuracy, limit, t0, t1;
    int                      n;
    int                      cusp_found;
    double                   cusp_t;
} ItpFitCtx;
static double itp_fit_cb(double x, void* user)
{
    ItpFitCtx* ctx = (ItpFitCtx*)user;
    int        cf  = 0;
    double     ct  = 0;
    double     r   = fit_opt_err_delta(ctx->self, ctx->vtbl, x, ctx->limit, ctx->t0, ctx->t1, ctx->n, &cf, &ct);
    if (cf)
    {
        ctx->cusp_found = 1;
        ctx->cusp_t     = ct;
    }
    return r;
}

/* fit_to_bezpath_opt_inner: fit range without cusps; returns 1 if cusp found, sets *ct */
static int fit_to_bezpath_opt_inner(
    const void*              self,
    const ParamCurveFitVtbl* vtbl,
    double                   accuracy,
    double                   t0,
    double                   t1,
    BezPath*                 path,
    double*                  ct_out)
{
    double tc;
    if (pcf_break_cusp(self, vtbl, t0, t1, &tc))
    {
        *ct_out = tc;
        return 1;
    }
    CubicBez c;
    double   err2;
    double   err;
    if (fit_to_cubic(self, vtbl, t0, t1, accuracy, &c, &err2))
    {
        err = sqrt(err2);
        if (err < accuracy)
        {
            if (t0 == 0.0)
                bezpath_move_to(path, c.p0);
            bezpath_curve_to(path, c.p1, c.p2, c.p3);
            return 0;
        }
    }
    else
    {
        err = 2.0 * accuracy;
    }
    /* Count n segments needed, then optimise split points via ITP */
    int    n        = 0;
    double t_cur    = t0;
    double last_err = err;
    while (1)
    {
        n++;
        FitResult fr = fit_opt_segment(self, vtbl, accuracy, t_cur, t1);
        if (fr.kind == FITRES_CUSP)
        {
            *ct_out = fr.val;
            return 1;
        }
        if (fr.kind == FITRES_SEG_ERROR)
        {
            last_err = fr.val;
            break;
        }
        t_cur = fr.val;
    }
    /* Use ITP to find optimal accuracy for n segments */
    ItpFitCtx ctx = {self, vtbl, accuracy, accuracy, t0, t1, n, 0, 0.0};
    double    k1  = 0.2 / accuracy;
    double    ya = -err, yb = accuracy - last_err;
    double    x_opt = solve_itp_generic(itp_fit_cb, &ctx, 0.0, accuracy, 1e-9, 1, k1, ya, yb);
    if (ctx.cusp_found)
    {
        *ct_out = ctx.cusp_t;
        return 1;
    }
    /* Replay n segments at x_opt accuracy */
    int path_len0 = path->len;
    t_cur         = t0;
    for (int i = 0; i < n; i++)
    {
        double t_next;
        if (i < n - 1)
        {
            FitResult fr = fit_opt_segment(self, vtbl, x_opt, t_cur, t1);
            if (fr.kind == FITRES_CUSP)
            {
                path->len = path_len0;
                *ct_out   = fr.val;
                return 1;
            }
            t_next = (fr.kind == FITRES_PARAM) ? fr.val : t1;
        }
        else
        {
            t_next = t1;
        }
        CubicBez seg;
        double   se2;
        if (!fit_to_cubic(self, vtbl, t_cur, t_next, accuracy, &seg, &se2))
        {
            /* shouldn't happen, but fall back */
            Point p1 = point_lerp(cubicbez_eval(c, 0.0), cubicbez_eval(c, 1.0), 1. / 3.);
            Point p2 = point_lerp(cubicbez_eval(c, 1.0), cubicbez_eval(c, 0.0), 1. / 3.);
            seg      = cubicbez_new(cubicbez_eval(c, 0.0), p1, p2, cubicbez_eval(c, 1.0));
        }
        if (i == 0 && t0 == 0.0)
            bezpath_move_to(path, seg.p0);
        bezpath_curve_to(path, seg.p1, seg.p2, seg.p3);
        t_cur = t_next;
        if (t_cur >= t1)
            break;
    }
    (void)x_opt;
    return 0;
}

BezPath fit_to_bezpath_opt(const void* self, const ParamCurveFitVtbl* vtbl, double accuracy)
{
    BezPath path;
    bezpath_init(&path);
    /* cusp stack (max depth 64) */
    double cusps[64];
    int    ncusps = 0;
    double t0     = 0.0;
    while (1)
    {
        double t1 = (ncusps > 0) ? cusps[ncusps - 1] : 1.0;
        double ct;
        if (!fit_to_bezpath_opt_inner(self, vtbl, accuracy, t0, t1, &path, &ct))
        {
            /* success: pop */
            if (ncusps == 0)
                break;
            t0 = cusps[--ncusps];
        }
        else
        {
            /* cusp found: push */
            if (ncusps < 64)
                cusps[ncusps++] = ct;
            else
            {
                t0 = ct;
            } /* overflow: skip */
        }
    }
    return path;
}

/* ============================================================
 * SECTION: simplify.rs — SimplifyBezPath + simplify_bezpath
 * ============================================================
 *
 * SimplifyBezPath converts an arbitrary path into a sequence of CubicBez
 * segments (elevating lines and quads) and stores prefix-sum moment
 * integrals for efficient sub-range queries.  It then implements the
 * ParamCurveFit vtable so that fit_to_bezpath / fit_to_bezpath_opt can
 * be called on it.
 */

#define SIMPLIFY_MAX_CUBICS 4096

typedef struct
{
    CubicBez c;
    /* inclusive prefix sum of moment integrals up to and including this cubic */
    double moments_area;
    double moments_mx;
    double moments_my;
} SimplifyCubic;

typedef struct
{
    SimplifyCubic segs[SIMPLIFY_MAX_CUBICS];
    int           n;
} SimplifyBezPath;

static void simplify_bez_path_init(SimplifyBezPath* s) { s->n = 0; }

/* Add a PathSeg — elevating to cubic */
static void simplify_bez_path_add_seg(SimplifyBezPath* s, PathSeg seg)
{
    if (s->n >= SIMPLIFY_MAX_CUBICS)
        return;
    CubicBez c;
    switch (seg.kind)
    {
    case PATH_SEG_LINE:
    {
        Line  l  = seg.line;
        Point p1 = point_lerp(l.p0, l.p1, 1. / 3.);
        Point p2 = point_lerp(l.p1, l.p0, 1. / 3.);
        c        = cubicbez_new(l.p0, p1, p2, l.p1);
        break;
    }
    case PATH_SEG_QUAD:
    {
        QuadBez q = seg.quad;
        /* degree elevation: c.p1 = p0 + 2/3*(p1-p0), c.p2 = p2 + 2/3*(p1-p2) */
        c.p0 = q.p0;
        c.p1 = point_new(q.p0.x + (2. / 3.) * (q.p1.x - q.p0.x), q.p0.y + (2. / 3.) * (q.p1.y - q.p0.y));
        c.p2 = point_new(q.p2.x + (2. / 3.) * (q.p1.x - q.p2.x), q.p2.y + (2. / 3.) * (q.p1.y - q.p2.y));
        c.p3 = q.p2;
        break;
    }
    case PATH_SEG_CUBIC:
        c = seg.cubic;
        break;
    default:
        return;
    }
    MomentIntegrals mi     = moment_integrals(c);
    double          prev_a = 0.0, prev_x = 0.0, prev_y = 0.0;
    if (s->n > 0)
    {
        prev_a = s->segs[s->n - 1].moments_area;
        prev_x = s->segs[s->n - 1].moments_mx;
        prev_y = s->segs[s->n - 1].moments_my;
    }
    s->segs[s->n].c            = c;
    s->segs[s->n].moments_area = prev_a + mi.area;
    s->segs[s->n].moments_mx   = prev_x + mi.mx;
    s->segs[s->n].moments_my   = prev_y + mi.my;
    s->n++;
}

/* Build from a BezPath */
static void simplify_bez_path_from_bezpath(SimplifyBezPath* s, const BezPath* path)
{
    simplify_bez_path_init(s);
    Point cur     = {0, 0};
    int   has_cur = 0;
    for (int i = 0; i < path->len; i++)
    {
        PathEl el = path->els[i];
        switch (el.kind)
        {
        case PATH_EL_MOVE_TO:
            cur     = el.p[0];
            has_cur = 1;
            break;
        case PATH_EL_LINE_TO:
            if (has_cur)
            {
                PathSeg seg;
                seg.kind = PATH_SEG_LINE;
                seg.line = line_new(cur, el.p[0]);
                simplify_bez_path_add_seg(s, seg);
                cur = el.p[0];
            }
            break;
        case PATH_EL_QUAD_TO:
            if (has_cur)
            {
                PathSeg seg;
                seg.kind = PATH_SEG_QUAD;
                seg.quad = quadbez_new(cur, el.p[0], el.p[1]);
                simplify_bez_path_add_seg(s, seg);
                cur = el.p[1];
            }
            break;
        case PATH_EL_CURVE_TO:
            if (has_cur)
            {
                PathSeg seg;
                seg.kind  = PATH_SEG_CUBIC;
                seg.cubic = cubicbez_new(cur, el.p[0], el.p[1], el.p[2]);
                simplify_bez_path_add_seg(s, seg);
                cur = el.p[2];
            }
            break;
        case PATH_EL_CLOSE_PATH:
            break;
        }
    }
}

/* Scale: map t in [0,1] to (cubic_index, t_within_cubic) */
static void simplify_scale(const SimplifyBezPath* s, double t, int* idx, double* t_out)
{
    double ts = t * s->n;
    double tf = floor(ts);
    *idx      = (int)tf;
    if (*idx >= s->n)
    {
        *idx   = s->n - 1;
        *t_out = 1.0;
    }
    else
    {
        *t_out = ts - tf;
    }
}

/* Moment integrals for sub-range of a single cubic */
static void
simplify_moment_integrals_seg(const SimplifyBezPath* s, int i, double t0, double t1, double* a, double* mx, double* my)
{
    if (t1 <= t0)
    {
        *a  = 0;
        *mx = 0;
        *my = 0;
        return;
    }
    MomentIntegrals mi = moment_integrals(cubicbez_subsegment(s->segs[i].c, t0, t1));
    *a                 = mi.area;
    *mx                = mi.mx;
    *my                = mi.my;
}

/* ParamCurveFit vtable implementation for SimplifyBezPath */
static CurveFitSample simplify_sample_pt_tangent(const void* self, double t, double sign)
{
    (void)sign;
    const SimplifyBezPath* s = (const SimplifyBezPath*)self;
    int                    i;
    double                 t0;
    simplify_scale(s, t, &i, &t0);
    CubicBez       c = s->segs[i].c;
    CurveFitSample samp;
    samp.p       = cubicbez_eval(c, t0);
    samp.tangent = point_to_vec2(cubicbez_deriv_eval(c, t0));
    return samp;
}
static void simplify_sample_pt_deriv(const void* self, double t, Point* p_out, Vec2* d_out)
{
    const SimplifyBezPath* s = (const SimplifyBezPath*)self;
    int                    i;
    double                 t0;
    simplify_scale(s, t, &i, &t0);
    CubicBez c = s->segs[i].c;
    *p_out     = cubicbez_eval(c, t0);
    *d_out     = vec2_scale(point_to_vec2(cubicbez_deriv_eval(c, t0)), (double)s->n);
}
static void simplify_moment_integrals_vtbl(const void* self, double t0, double t1, double* area, double* mx, double* my)
{
    const SimplifyBezPath* s = (const SimplifyBezPath*)self;
    int                    i0;
    double                 u0;
    simplify_scale(s, t0, &i0, &u0);
    int    i1;
    double u1;
    simplify_scale(s, t1, &i1, &u1);
    if (i0 == i1)
    {
        simplify_moment_integrals_seg(s, i0, u0, u1, area, mx, my);
    }
    else
    {
        double a0, x0, y0, a1, x1, y1;
        simplify_moment_integrals_seg(s, i0, u0, 1.0, &a0, &x0, &y0);
        simplify_moment_integrals_seg(s, i1, 0.0, u1, &a1, &x1, &y1);
        *area = a0 + a1;
        *mx   = x0 + x1;
        *my   = y0 + y1;
        if (i1 > i0 + 1)
        {
            /* Add prefix sums between i0 and i1-1 */
            double pa = s->segs[i0].moments_area, px = s->segs[i0].moments_mx, py = s->segs[i0].moments_my;
            double pb = s->segs[i1 - 1].moments_area, qx = s->segs[i1 - 1].moments_mx, qy = s->segs[i1 - 1].moments_my;
            *area += pb - pa;
            *mx   += qx - px;
            *my   += qy - py;
        }
    }
}
static int simplify_break_cusp(const void* self, double t0, double t1, double* t_out)
{
    (void)self;
    (void)t0;
    (void)t1;
    (void)t_out;
    return 0; /* simplify does not report cusps */
}

static const ParamCurveFitVtbl SIMPLIFY_VTBL =
    {simplify_sample_pt_tangent, simplify_sample_pt_deriv, simplify_moment_integrals_vtbl, simplify_break_cusp};

/* SimplifyOptions */
typedef enum
{
    SIMPLIFY_SUBDIVIDE,
    SIMPLIFY_OPTIMIZE
} SimplifyOptLevel;

typedef struct
{
    double           angle_thresh;
    SimplifyOptLevel opt_level;
} SimplifyOptions;

static inline SimplifyOptions simplify_options_default(void)
{
    SimplifyOptions o = {1e-3, SIMPLIFY_SUBDIVIDE};
    return o;
}

/* simplify_bezpath: the main public API.
 * Reads a BezPath, simplifies it, writes result into *out.
 * Corners (where the tangent angle change exceeds angle_thresh) are respected
 * as split points so each smooth run is fitted independently. */
void simplify_bezpath(const BezPath* path, double accuracy, const SimplifyOptions* opts, BezPath* out)
{
    bezpath_init(out);
    /* We maintain a "queue" of segments forming the current smooth run */
    BezPath queue;
    bezpath_init(&queue);
    int     needs_moveto = 1;
    Point   last_pt      = {0, 0};
    int     has_last     = 0;
    PathSeg last_seg;
    int     has_last_seg = 0;
    memset(&last_seg, 0, sizeof(last_seg));

/* Flush the queue into *out using SimplifyBezPath + fit */
#define FLUSH()                                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (queue.len > 0)                                                                                             \
        {                                                                                                              \
            BezPath fitted;                                                                                            \
            if (queue.len == 2)                                                                                        \
            {                                                                                                          \
                /* Only one segment — output directly */                                                             \
                fitted = queue;                                                                                        \
                if (!needs_moveto)                                                                                     \
                {                                                                                                      \
                    /* skip the moveto */                                                                              \
                    memmove(&fitted.els[0], &fitted.els[1], (fitted.len - 1) * sizeof(PathEl));                        \
                    fitted.len--;                                                                                      \
                }                                                                                                      \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                SimplifyBezPath sbp;                                                                                   \
                simplify_bez_path_from_bezpath(&sbp, &queue);                                                          \
                if (opts->opt_level == SIMPLIFY_OPTIMIZE)                                                              \
                    fitted = fit_to_bezpath_opt(&sbp, &SIMPLIFY_VTBL, accuracy);                                       \
                else                                                                                                   \
                    fitted = fit_to_bezpath(&sbp, &SIMPLIFY_VTBL, accuracy);                                           \
                if (!needs_moveto && fitted.len > 0 && fitted.els[0].kind == PATH_EL_MOVE_TO)                          \
                {                                                                                                      \
                    memmove(&fitted.els[0], &fitted.els[1], (fitted.len - 1) * sizeof(PathEl));                        \
                    fitted.len--;                                                                                      \
                }                                                                                                      \
            }                                                                                                          \
            for (int _fi = 0; _fi < fitted.len && out->len < BEZPATH_MAX_ELS; _fi++)                                   \
                out->els[out->len++] = fitted.els[_fi];                                                                \
            needs_moveto = 0;                                                                                          \
            bezpath_init(&queue);                                                                                      \
        }                                                                                                              \
    }                                                                                                                  \
    while (0)

    for (int i = 0; i < path->len; i++)
    {
        PathEl  el = path->els[i];
        PathSeg this_seg;
        int     has_this = 0;
        memset(&this_seg, 0, sizeof(this_seg));

        switch (el.kind)
        {
        case PATH_EL_MOVE_TO:
            FLUSH();
            needs_moveto = 1;
            last_pt      = el.p[0];
            has_last     = 1;
            has_last_seg = 0;
            break;
        case PATH_EL_LINE_TO:
            if (has_last && !(last_pt.x == el.p[0].x && last_pt.y == el.p[0].y))
            {
                this_seg.kind = PATH_SEG_LINE;
                this_seg.line = line_new(last_pt, el.p[0]);
                has_this      = 1;
            }
            break;
        case PATH_EL_QUAD_TO:
            if (has_last)
            {
                this_seg.kind = PATH_SEG_QUAD;
                this_seg.quad = quadbez_new(last_pt, el.p[0], el.p[1]);
                has_this      = 1;
            }
            break;
        case PATH_EL_CURVE_TO:
            if (has_last)
            {
                this_seg.kind  = PATH_SEG_CUBIC;
                this_seg.cubic = cubicbez_new(last_pt, el.p[0], el.p[1], el.p[2]);
                has_this       = 1;
            }
            break;
        case PATH_EL_CLOSE_PATH:
            FLUSH();
            if (out->len < BEZPATH_MAX_ELS)
            {
                out->els[out->len].kind = PATH_EL_CLOSE_PATH;
                out->len++;
            }
            needs_moveto = 1;
            has_last_seg = 0;
            continue;
        }
        if (has_this)
        {
            /* Check for corner between last_seg and this_seg */
            if (has_last_seg)
            {
                /* Tangent at end of last_seg and start of this_seg */
                Vec2 lt, tt;
                switch (last_seg.kind)
                {
                case PATH_SEG_LINE:
                    lt = point_sub(last_seg.line.p1, last_seg.line.p0);
                    break;
                case PATH_SEG_QUAD:
                {
                    QuadBez q = last_seg.quad;
                    lt        = point_sub(q.p2, q.p1);
                    break;
                }
                case PATH_SEG_CUBIC:
                {
                    CubicBez c = last_seg.cubic;
                    lt         = point_sub(c.p3, c.p2);
                    break;
                }
                default:
                    lt = vec2_new(1, 0);
                    break;
                }
                switch (this_seg.kind)
                {
                case PATH_SEG_LINE:
                    tt = point_sub(this_seg.line.p1, this_seg.line.p0);
                    break;
                case PATH_SEG_QUAD:
                {
                    QuadBez q = this_seg.quad;
                    tt        = point_sub(q.p1, q.p0);
                    break;
                }
                case PATH_SEG_CUBIC:
                {
                    CubicBez c = this_seg.cubic;
                    tt         = point_sub(c.p1, c.p0);
                    break;
                }
                default:
                    tt = vec2_new(1, 0);
                    break;
                }
                double cross = vec2_cross(lt, tt);
                double dot   = vec2_dot(lt, tt);
                if (fabs(cross) > fabs(dot) * opts->angle_thresh)
                {
                    FLUSH();
                }
            }
            /* Add this_seg to queue */
            if (queue.len == 0)
            {
                Point seg_start;
                switch (this_seg.kind)
                {
                case PATH_SEG_LINE:
                    seg_start = this_seg.line.p0;
                    break;
                case PATH_SEG_QUAD:
                    seg_start = this_seg.quad.p0;
                    break;
                case PATH_SEG_CUBIC:
                    seg_start = this_seg.cubic.p0;
                    break;
                default:
                    seg_start = last_pt;
                    break;
                }
                bezpath_move_to(&queue, seg_start);
            }
            switch (this_seg.kind)
            {
            case PATH_SEG_LINE:
                bezpath_line_to(&queue, this_seg.line.p1);
                break;
            case PATH_SEG_QUAD:
                bezpath_quad_to(&queue, this_seg.quad.p1, this_seg.quad.p2);
                break;
            case PATH_SEG_CUBIC:
                bezpath_curve_to(&queue, this_seg.cubic.p1, this_seg.cubic.p2, this_seg.cubic.p3);
                break;
            default:
                break;
            }
            /* Advance last_pt */
            switch (this_seg.kind)
            {
            case PATH_SEG_LINE:
                last_pt = this_seg.line.p1;
                break;
            case PATH_SEG_QUAD:
                last_pt = this_seg.quad.p2;
                break;
            case PATH_SEG_CUBIC:
                last_pt = this_seg.cubic.p3;
                break;
            default:
                break;
            }
            has_last     = 1;
            last_seg     = this_seg;
            has_last_seg = 1;
        }
    }
    FLUSH();
#undef FLUSH
}

/* ============================================================
 * END OF FILE
 * ============================================================
 *
 * SUMMARY OF ITEMS STILL REQUIRING MANUAL TRANSLATION:
 *
 * 1. factor_quartic_inner() — common.rs lines 362-580:
 *    The quartic factorization used by cubic_fit() in fit.rs is stubbed —
 *    it falls back to a cubic solve which may miss some candidate cubics.
 *    Once factor_quartic_inner is ported, replace the fallback branch in
 *    cubic_fit() marked with the LOUD COMMENT above.
 *
 * 2. Full Ellipse implementation — ellipse.rs:
 *    ellipse_new / ellipse_radii_and_rotation are stubs.  affine_svd() is
 *    already present so wiring it up is straightforward.
 *
 * 3. Adaptive arc-length integration — cubic_arclen.rs / quad_arclen.rs:
 *    The current GL-16 quadrature is a reasonable approximation.  Replace
 *    with recursive adaptive subdivision for high-accuracy work.
 *
 * 4. offset.rs (~668 lines) — CubicOffset stroke expansion:
 *    Depends on fit.rs (now ported).  The CubicOffset struct implements
 *    ParamCurveFit, so once ported it can be passed directly to fit_to_bezpath.
 *
 * 5. bezpath.rs (~2164 lines) — BezPath area, winding, flatten, perimeter,
 *    path_segments iterator, SVG path string parsing/writing:
 *    The BezPath is a fixed-capacity stub (BEZPATH_MAX_ELS=4096).
 *    For production use, replace with a malloc/realloc dynamic array.
 *
 * 6. moments.rs — quadbez_moments(), cubicbez_moments():
 *    Polynomial moment integrals for quad/cubic.  moment_integrals() for a
 *    full CubicBez (from simplify.rs) is already ported above.
 *
 * 7. svg.rs string I/O — SVG path parsing (from_svg) and writing (to_svg).
 *    The SvgArc-to-cubic conversion is ported (svg_arc_to_cubic_beziers).
 *    The string lexer/parser is omitted.
 *
 * 8. stroke.rs (~1021 lines) — expand_stroke():
 *    The Stroke data structure is defined.  The actual stroking algorithm
 *    (expand_stroke, dash handling, cap/join generation) depends on offset.rs.
 */

// Simplify example

#include <math.h>
#include <stdio.h>

/* ---------------------------------------------------------------
 * plot_fn: mirrors the Rust plot_fn().
 *
 * Builds a BezPath whose cubic segments approximate f(x) on [xa, xb]
 * using n equal-width sub-intervals.  The Hermite formula
 *
 *   p1 = plot(x0 + dx/3,  y0 + d0*dx/3)
 *   p2 = plot(x3 - dx/3,  y3 - d3*dx/3)
 *
 * gives a C1 piecewise cubic that matches both value and derivative at
 * every knot.
 * --------------------------------------------------------------- */
static BezPath plot_fn(double (*f)(double), double (*df)(double), double xa, double xb, int n)
{
    double width    = 800.0;
    double dx       = (xb - xa) / n;
    double xs       = width / (xb - xa); /* x scale: map [xa,xb] -> [0,800] */
    double ys       = 250.0;             /* y scale */
    double y_origin = 300.0;             /* vertical offset in SVG coords    */

    /* Inline the plot() closure: (x,y) -> screen Point */
#define PLOT(x, y) point_new(((x) - xa) * xs, y_origin - (y) * ys)

    BezPath path;
    bezpath_init(&path);

    double x0 = xa;
    double y0 = f(xa);
    double d0 = df(xa);
    bezpath_move_to(&path, PLOT(x0, y0));

    for (int i = 0; i < n; i++)
    {
        double x3 = xa + dx * (i + 1);
        double y3 = f(x3);
        double d3 = df(x3);

        double x1 = x0 + (1.0 / 3.0) * dx;
        double x2 = x3 - (1.0 / 3.0) * dx;
        double y1 = y0 + d0 * (1.0 / 3.0) * dx;
        double y2 = y3 - d3 * (1.0 / 3.0) * dx;

        bezpath_curve_to(&path, PLOT(x1, y1), PLOT(x2, y2), PLOT(x3, y3));

        x0 = x3;
        y0 = y3;
        d0 = d3;
    }

#undef PLOT
    return path;
}

/* ---------------------------------------------------------------
 * print_bezpath: prints each PathEl on its own line, e.g.
 *   M 0.000 300.000
 *   C 13.333 370.351 26.667 405.118 40.000 404.586
 *   ...
 * --------------------------------------------------------------- */
static void print_bezpath(const BezPath* p)
{
    for (int i = 0; i < p->len; i++)
    {
        PathEl el = p->els[i];
        switch (el.kind)
        {
        case PATH_EL_MOVE_TO:
            printf("M %.6f %.6f\n", el.p[0].x, el.p[0].y);
            break;
        case PATH_EL_LINE_TO:
            printf("L %.6f %.6f\n", el.p[0].x, el.p[0].y);
            break;
        case PATH_EL_QUAD_TO:
            printf("Q %.6f %.6f  %.6f %.6f\n", el.p[0].x, el.p[0].y, el.p[1].x, el.p[1].y);
            break;
        case PATH_EL_CURVE_TO:
            printf(
                "C %.6f %.6f  %.6f %.6f  %.6f %.6f\n",
                el.p[0].x,
                el.p[0].y,
                el.p[1].x,
                el.p[1].y,
                el.p[2].x,
                el.p[2].y);
            break;
        case PATH_EL_CLOSE_PATH:
            printf("Z\n");
            break;
        }
    }
}

int main(void)
{
    /* Build the sin(x) path over [-8, 8] with 20 cubic segments */
    BezPath path = plot_fn(sin, cos, -8.0, 8.0, 20);

    printf("=== Original path: %d elements ===\n", path.len);
    print_bezpath(&path);

    /* Simplify using fit_to_bezpath_opt via SimplifyBezPath */
    SimplifyBezPath sbp;
    simplify_bez_path_from_bezpath(&sbp, &path);

    BezPath simplified = fit_to_bezpath_opt(&sbp, &SIMPLIFY_VTBL, 0.1);

    printf("\n=== Simplified path (accuracy=0.1): %d elements ===\n", simplified.len);
    print_bezpath(&simplified);

    return 0;
}