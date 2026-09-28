#include <assert.h>
#define _USE_MATH_DEFINES /* needed for MS Windows to define M_PI */
#include <math.h>
#include <stdlib.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION
#endif

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cdrizzle_blot_api

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "driz_portability.h"
#include "cdrizzlemap.h"
#include "cdrizzleblot.h"
#include "cdrizzleutil.h"

static const double lut_delta = 0.003; /* spacing of Lanczos LUT */

/** ---------------------------------------------------------------------------
 * Signature for functions that perform blotting interpolation.
 */

typedef int(interp_function)(
    const void *, const struct image *, const double, const double,
    /* Output parameters */
    double *, struct driz_error_t *);

/** ---------------------------------------------------------------------------
 * A standard set of asserts for all of the interpolation functions
 */

#define INTERPOLATION_ASSERTS                  \
    assert(data);                              \
    assert(isize[0] > 0);                      \
    assert(isize[1] > 0);                      \
    assert(x >= 0.0 && x < (double) isize[0]); \
    assert(y >= 0.0 && y < (double) isize[1]); \
    assert(value);                             \
    assert(error);

/** ---------------------------------------------------------------------------
 * Procedure to evaluate the bicubic polynomial interpolant.  The array coeff
 * contains the coefficients of the 2D interpolant.  The procedure assumes that
 * 1 <= x <= isize[0] and 1 <= y <= isize[1] and that coeff[1+first_point] =
 * datain[1,1]. The interpolant is evaluated using Everett's central difference
 * formula. (Was: IIBIP3)
 *
 * coeff:     Array of shape \a [len_coeff][len_coeff] contains the coefficients
 * of the 2D interpolant. len_coeff: The dimension (each side of the square) of
 * the coefficient array. firstt:    Offset of the first data point.  (In
 * practice, this is always zero.) npts:      The number of points to calculate.
 * x:         An array of length \a npts of x values.
 * y:         An array of length \a npts of y values.
 * zfit:      An array of length \a npts of interpolated values. (output)
 */

static inline_macro int
ii_bipoly3(
    const double *coeff /* [len_coeff][len_coeff] */, const integer_t len_coeff,
    const integer_t firstt, const integer_t npts, const double *x /* [npts] */,
    const double *y /* [npts] */,
    /* Output parameters */
    double *zfit /* [npts] */, struct driz_error_t *error)
{
    double sx, tx, sx2m1, tx2m1, sy, ty;
    double cd20[4], cd21[4], ztemp[4];
    double cd20y, cd21y;
    integer_t nxold, nyold;
    integer_t nx, ny;
    integer_t firstw, index;
    integer_t i, j;

    nxold = nyold = -1;

    for (i = 0; i < npts; ++i) {
        nx = (integer_t) x[i];
        ny = (integer_t) y[i];
        if (nx < 0 || ny < 0) {
            driz_error_set(error, PyExc_ValueError, "Negative coordinates in ii_bipoly3.");
            return 1;
        }

        sx = x[i] - (double) nx;
        tx = 1.0 - sx;
        sx2m1 = sx * sx - 1.0;
        tx2m1 = tx * tx - 1.0;
        sy = y[i] - (double) ny;
        ty = 1.0 - sy;

        /* Calculate pointer to data[nx, ny-1] */
        firstw = firstt + (ny - 2) * len_coeff + nx - 1;

        /* loop over the 4 surrounding rows of data calculate the central
           differences at each value of y

           If new data point calculate the central differences in x for
           each y */
        if (nx != nxold || ny != nyold) {
            for (j = 0, index = firstw; j < 4; ++j, index += len_coeff) {
                assert(index > 0 && index < (len_coeff * len_coeff) - 2);

                cd20[j] = 1.0 / 6.0 * (coeff[index + 1] - 2.0 * coeff[index] + coeff[index - 1]);
                cd21[j] = 1.0 / 6.0 * (coeff[index + 2] - 2.0 * coeff[index + 1] + coeff[index]);
            }
        }

        /* Interpolate in x at each value of y */
        for (j = 0, index = firstw; j < 4; ++j, index += len_coeff) {
            assert(index >= 0 && index < (len_coeff * len_coeff) - 1);

            ztemp[j] =
                sx * (coeff[index + 1] + sx2m1 * cd21[j]) + tx * (coeff[index] + tx2m1 * cd20[j]);
        }

        /* Calculate y central differences */
        cd20y = 1.0 / 6.0 * (ztemp[2] - 2.0 * ztemp[1] + ztemp[0]);
        cd21y = 1.0 / 6.0 * (ztemp[3] - 2.0 * ztemp[2] + ztemp[1]);

        /* Interpolate in y */
        zfit[i] =
            sy * (ztemp[2] + (sy * sy - 1.0) * cd21y) + ty * (ztemp[1] + (ty * ty - 1.0) * cd20y);

        nxold = nx;
        nyold = ny;
    }
    return 0;
}

/** ---------------------------------------------------------------------------
 * Procedure to evaluate a biquintic polynomial.  The array coeff contains the
 * coefficients of the 2D interpolant.  The routine assumes that 0 <= x <
 * isize[0] and 0 <= y < isize[1]. The interpolant is evaluated using Everett's
 * central difference formula. (Was: IIBIP5)
 *
 * coeff:     Array of shape \a [len_coeff][len_coeff] contains the coefficients
 * of the 2D interpolant. len_coeff: The dimension (each side of the square) of
 * the coefficient array. firstt:    Offset of the first data point.  (In
 * practice, this is always zero.) npts:      The number of points to calculate.
 * x:         An array of length \a npts of x values.
 * y:         An array of length \a npts of y values.
 * zfit:      An array of length \a npts of interpolated values. (output)
 */

static inline_macro int
ii_bipoly5(
    const double *coeff /* [len_coeff][len_coeff] */, const integer_t len_coeff,
    const integer_t firstt, const integer_t npts, const double *x /* [npts] */,
    const double *y /* [npts] */,
    /* Output parameters */
    double *zfit /* [npts] */, struct driz_error_t *error)
{
    integer_t nxold, nyold;
    integer_t nx, ny;
    double sx, sx2, tx, tx2, sy, sy2, ty, ty2;
    double sx2m1, sx2m4, tx2m1, tx2m4;
    double cd20[6], cd21[6], cd40[6], cd41[6];
    double cd20y, cd21y, cd40y, cd41y;
    double ztemp[6];
    integer_t firstw, index;
    integer_t i, j;

    assert(coeff);
    assert(len_coeff > 0);
    assert(npts > 0);
    assert(x);
    assert(y);
    assert(zfit);

    nxold = nyold = -1;

    for (i = 0; i < npts; ++i) {
        nx = (integer_t) x[i];
        ny = (integer_t) y[i];
        if (nx < 0 || ny < 0) {
            driz_error_set(error, PyExc_ValueError, "Negative coordinates in ii_bipoly3.");
            return 1;
        }

        sx = x[i] - (double) nx;
        sx2 = sx * sx;
        sx2m1 = sx2 - 1.0;
        sx2m4 = sx2 - 4.0;
        tx = 1.0 - sx;
        tx2 = tx * tx;
        tx2m1 = tx2 - 1.0;
        tx2m4 = tx2 - 4.0;

        sy = y[i] - (double) ny;
        sy2 = sy * sy;
        ty = 1.0 - sy;
        ty2 = ty * ty;

        /* Calculate value of pointer to data[nx,ny-2] */
        firstw = firstt + (ny - 3) * len_coeff + nx - 1;

        /* Calculate the central differences in x at each value of y */
        if (nx != nxold || ny != nyold) {
            for (j = 0, index = firstw; j < 6; ++j, index += len_coeff) {
                assert(index >= 1 && index < len_coeff * len_coeff - 3);

                cd20[j] = 1.0 / 6.0 * (coeff[index + 1] - 2.0 * coeff[index] + coeff[index - 1]);
                cd21[j] = 1.0 / 6.0 * (coeff[index + 2] - 2.0 * coeff[index + 1] + coeff[index]);
                cd40[j] = 1.0 / 120.0 *
                          (coeff[index - 2] - 4.0 * coeff[index - 1] + 6.0 * coeff[index] -
                           4.0 * coeff[index + 1] + coeff[index + 2]);
                cd41[j] = 1.0 / 120.0 *
                          (coeff[index - 1] - 4.0 * coeff[index] + 6.0 * coeff[index + 1] -
                           4.0 * coeff[index + 2] + coeff[index + 3]);
            }
        }

        /* Interpolate in x at each value of y */
        for (j = 0, index = firstw; j < 6; ++j, index += len_coeff) {
            assert(index >= 0 && index < len_coeff * len_coeff - 1);

            ztemp[j] = sx * (coeff[index + 1] + sx2m1 * (cd21[j] + sx2m4 * cd41[j])) +
                       tx * (coeff[index] + tx2m1 * (cd20[j] + tx2m4 * cd40[j]));
        }

        /* Central differences in y */
        cd20y = 1.0 / 6.0 * (ztemp[3] - 2.0 * ztemp[2] + ztemp[1]);
        cd21y = 1.0 / 6.0 * (ztemp[4] - 2.0 * ztemp[3] + ztemp[2]);
        cd40y =
            1.0 / 120.0 * (ztemp[0] - 4.0 * ztemp[1] + 6.0 * ztemp[2] - 4.0 * ztemp[3] + ztemp[4]);
        cd41y =
            1.0 / 120.0 * (ztemp[1] - 4.0 * ztemp[2] + 6.0 * ztemp[3] - 4.0 * ztemp[4] + ztemp[5]);

        /* Interpolate in y */
        zfit[i] = sy * (ztemp[3] + (sy2 - 1.0) * (cd21y + (sy2 - 4.0) * cd41y)) +
                  ty * (ztemp[2] + (ty2 - 1.0) * (cd20y + (ty2 - 4.0) * cd40y));

        nxold = nx;
        nyold = ny;
    }
    return 0;
}

/** ---------------------------------------------------------------------------
 * Perform nearest neighbor interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type.
 * (NULL). data:  A 2D data array x:     The fractional x coordinate y:     The
 * fractional y coordinate value: The resulting value at x, y after
 * interpolating the data (output) error: The error structure (output)
 */

static int
interpolate_nearest_neighbor(
    const void *state, const struct image *data, const double x, const double y,
    /* Output parameters */
    double *value, struct driz_error_t *error)
{
    /* Unused parameters: */
    (void) state;
    (void) error;

    integer_t isize[2];
    isize[0] = data->size[0];
    isize[1] = data->size[1];

    assert(state == NULL);
    INTERPOLATION_ASSERTS;

    /* Points in the outer half of the last row or column round up to one
       past the end of the image, so clamp them to the last pixel. */
    *value = image_get(
        data, CLAMP_BELOW((integer_t) (x + 0.5), isize[0] - 1),
        CLAMP_BELOW((integer_t) (y + 0.5), isize[1] - 1));
    return 0;
}

/** ---------------------------------------------------------------------------
 * Perform basic bilinear interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type.
 * (NULL). data:  A 2D data array x:     The fractional x coordinate y:     The
 * fractional y coordinate value: The resulting value at x, y after
 * interpolating the data (output) error: The error structure (output)
 */

static int
interpolate_bilinear(
    const void *state, const struct image *data, const double x, const double y,
    /* Output parameters */
    double *value, struct driz_error_t *error)
{
    /* Unused parameters: */
    (void) state;
    (void) error;

    integer_t nx, ny;
    double sx, tx, sy, ty, f00;
    integer_t isize[2];

    isize[0] = data->size[0];
    isize[1] = data->size[1];

    assert(state == NULL);
    INTERPOLATION_ASSERTS;

    nx = (integer_t) x;
    ny = (integer_t) y;

    if (nx < 0 || ny < 0 || nx >= isize[0] || ny >= isize[1]) {
        driz_error_set_message(error, "Bilinear interpolation: point outside of the image.");
        return 1;
    }

    f00 = image_get(data, nx, ny);

    if (nx == (isize[0] - 1)) {
        if (ny == (isize[1] - 1)) {
            /* This is the last pixel (in x and y). Assign constant value of
             * this pixel. */
            *value = f00;
            return 0;
        }
        /* Interpolate along Y-direction only */
        sy = y - (double) ny;
        *value = (1.0 - sy) * f00 + sy * image_get(data, nx, ny + 1);
    } else if (ny == (isize[1] - 1)) {
        /* Interpolate along X-direction only */
        sx = x - (double) nx;
        *value = (1.0 - sx) * f00 + sx * image_get(data, nx + 1, ny);
    } else {
        /* Bilinear - interpolation */
        sx = x - (double) nx;
        tx = 1.0 - sx;
        sy = y - (double) ny;
        ty = 1.0 - sy;

        *value = tx * ty * f00 + sx * ty * image_get(data, nx + 1, ny) +
                 sy * tx * image_get(data, nx, ny + 1) + sx * sy * image_get(data, nx + 1, ny + 1);
    }

    return 0;
}

/** ---------------------------------------------------------------------------
 * Return the value of pixel (i, j), extending the image beyond its edges by
 * point reflection about the edge pixel:
 *
 *     data[-k]        = 2 * data[0]     - data[k]
 *     data[last + k]  = 2 * data[last]  - data[last - k]
 *
 * which keeps linear trends continuous across the edge. The reflection is
 * applied in x and then in y, so pixels beyond a corner are reflected in both
 * directions. The requested pixel must be less than isize pixels beyond an
 * edge so that the reflected pixel lies inside the image.
 */

static always_inline_macro double
get_pixel_reflect_x(
    const struct image *data, const integer_t isize[2], integer_t i, integer_t j,
    const enum image_types types)
{
    const integer_t last = isize[0] - 1;

    assert(i > -isize[0] && i < 2 * isize[0] - 1);

    if (i < 0) {
        return 2.0 * image_get_as(data, 0, j, types) - image_get_as(data, -i, j, types);
    } else if (i > last) {
        return 2.0 * image_get_as(data, last, j, types) -
               image_get_as(data, 2 * last - i, j, types);
    }
    return image_get_as(data, i, j, types);
}

static always_inline_macro double
get_pixel_reflect(
    const struct image *data, const integer_t isize[2], integer_t i, integer_t j,
    const enum image_types types)
{
    const integer_t last = isize[1] - 1;

    assert(j > -isize[1] && j < 2 * isize[1] - 1);

    if (j < 0) {
        return 2.0 * get_pixel_reflect_x(data, isize, i, 0, types) -
               get_pixel_reflect_x(data, isize, i, -j, types);
    } else if (j > last) {
        return 2.0 * get_pixel_reflect_x(data, isize, i, last, types) -
               get_pixel_reflect_x(data, isize, i, 2 * last - j, types);
    }
    return get_pixel_reflect_x(data, isize, i, j, types);
}

/** ---------------------------------------------------------------------------
 * Copy the n x n block of pixels starting at pixel (i0, j0) into block, as
 * block[row][col] = data[j0 + row, i0 + col], reflecting across the image
 * edges where the block extends beyond them.
 */

static always_inline_macro void
get_pixel_block_as(
    const struct image *data, const integer_t isize[2], const integer_t i0, const integer_t j0,
    const integer_t n, double *block /* [n][n] */, const enum image_types types)
{
    integer_t row, col;

    for (row = 0; row < n; ++row) {
        for (col = 0; col < n; ++col) {
            block[row * n + col] = get_pixel_reflect(data, isize, i0 + col, j0 + row, types);
        }
    }
}

/* get_pixel_block_as() with the type of the image checked once per block */
static inline_macro void
get_pixel_block(
    const struct image *data, const integer_t isize[2], const integer_t i0, const integer_t j0,
    const integer_t n, double *block /* [n][n] */)
{
    if (data->itemsize == sizeof(double)) {
        get_pixel_block_as(data, isize, i0, j0, n, block, images_float64);
    } else {
        get_pixel_block_as(data, isize, i0, j0, n, block, images_float32);
    }
}

/** ---------------------------------------------------------------------------
 * Perform cubic polynomial interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type.
 * (NULL). data:  A 2D data array x:     The fractional x coordinate y:     The
 * fractional y coordinate value: The resulting value at x, y after
 * interpolating the data (output) error: The error structure (output)
 */

static int
interpolate_poly3(
    const void *state, const struct image *data, const double x, const double y,
    /* Output parameters */
    double *value, struct driz_error_t *error)
{
    /* Unused parameters: */
    (void) state;

    integer_t nx, ny;
    const integer_t rowleh = 4;
    double coeff[4][4];
    double xval, yval;
    integer_t isize[2];
    isize[0] = data->size[0];
    isize[1] = data->size[1];

    assert(state == NULL);
    INTERPOLATION_ASSERTS;

    nx = (integer_t) x;
    ny = (integer_t) y;

    /* The 4x4 block of pixels around (x, y), with (nx, ny) at coeff[1][1] */
    get_pixel_block(data, isize, nx - 1, ny - 1, 4, &coeff[0][0]);

    xval = 2.0 + (x - (double) nx);
    yval = 2.0 + (y - (double) ny);

    return ii_bipoly3(&coeff[0][0], rowleh, 0, 1, &xval, &yval, value, error);
}

/** ---------------------------------------------------------------------------
 * Perform quintic polynomial interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type.
 * (NULL). data:  A 2D data array x:     The fractional x coordinate y:     The
 * fractional y coordinate value: The resulting value at x, y after
 * interpolating the data (output) error: The error structure (output)
 */

static int
interpolate_poly5(
    const void *state, const struct image *data, const double x, const double y,
    /* Output parameters */
    double *value, struct driz_error_t *error)
{
    (void) state;

    integer_t nx, ny;
    const integer_t rowleh = 6;
    double coeff[6][6];
    double xval, yval;
    integer_t isize[2];
    isize[0] = data->size[0];
    isize[1] = data->size[1];

    assert(state == NULL);
    INTERPOLATION_ASSERTS;

    nx = (integer_t) x;
    ny = (integer_t) y;

    /* The 6x6 block of pixels around (x, y), with (nx, ny) at coeff[2][2] */
    get_pixel_block(data, isize, nx - 2, ny - 2, 6, &coeff[0][0]);

    xval = 3.0 + (x - (double) nx);
    yval = 3.0 + (y - (double) ny);

    return ii_bipoly5(&coeff[0][0], rowleh, 0, 1, &xval, &yval, value, error);
}

/** ---------------------------------------------------------------------------
 * Perform Lanczos interpolation.
 *
 * state: A pointer to a \a lanczos_param_t object
 * data:  A 2D data array
 * x:     The fractional x coordinate
 * y:     The fractional y coordinate
 * value: The resulting value at x, y after interpolating the data (output)
 * error: The error structure (output)
 */

static int
interpolate_lanczos(
    const void *state, const struct image *data, const double x, const double y,
    /* Output parameters */
    double *value, struct driz_error_t *error)
{
    /* Unused parameters: */
    (void) error;

    integer_t ixs, iys, ixe, iye;
    size_t xoff, yoff;
    double luty, sum;
    double nbox;
    integer_t i, j;
    const struct lanczos_param_t *lanczos = (const struct lanczos_param_t *) state;
    integer_t isize[2];
    isize[0] = data->size[0];
    isize[1] = data->size[1];

    assert(state);
    INTERPOLATION_ASSERTS;

    nbox = (double) lanczos->nbox;

    /* First check for being close to the edge and, if so, return the
       missing value */
    ixs = MAX((integer_t) floor(x - nbox) + 1, 0);
    ixe = MIN((integer_t) floor(x + nbox), isize[0] - 1);
    iys = MAX((integer_t) floor(y - nbox) + 1, 0);
    iye = MIN((integer_t) floor(y + nbox), isize[1] - 1);

    /* Don't divide-by-zero errors */
    assert(lanczos->space != 0.0);

    /* Loop over the box, which is assumed to be scaled appropriately */
    sum = 0.0;
    for (j = iys; j <= iye; ++j) {
        yoff = (size_t) fabs(((double) y - (double) j) / lanczos->space);
        if (yoff >= lanczos->nlut) {
            continue;
        }

        luty = lanczos->lut[yoff];
        for (i = ixs; i <= ixe; ++i) {
            xoff = (size_t) (fabs(((double) x - (double) i) / lanczos->space));
            if (xoff >= lanczos->nlut) {
                continue;
            }

            sum += (double) image_get(data, i, j) * lanczos->lut[xoff] * luty;
        }
    }

    *value = sum;
    return 0;
}

/** ---------------------------------------------------------------------------
 * A mapping from e_interp_t enumeration values to function pointers that
 * actually perform the interpolation.  NULL elements will raise an
 * "unimplemented" error.
 */

interp_function *interp_function_map[interp_LAST] = {
    &interpolate_nearest_neighbor, &interpolate_bilinear, &interpolate_poly3, &interpolate_poly5,
    &interpolate_lanczos,          &interpolate_lanczos};

/** ---------------------------------------------------------------------------
 * Interpolate grid of pixels onto new grid of different size.
 *
 * p:   structure containing options, input, and output
 */

int
doblot(struct driz_param_t *p)
{
    int order;
    size_t nlut;
    integer_t isize[2], osize[2];
    double xo, yo, v, s;
    integer_t i, j;
    interp_function *interpolate;
    struct lanczos_param_t lanczos;
    void *state = NULL;

    driz_log_message("starting doblot");
    get_dimensions(p->data, isize);
    get_dimensions(p->output_data, osize);

    s = p->ef * p->iscale;

    /* Select interpolation function */
    assert(p->interpolation >= 0 && p->interpolation < interp_LAST);
    interpolate = interp_function_map[p->interpolation];
    if (interpolate == NULL) {
        driz_error_set_message(p->error, "Requested interpolation type not implemented.");
        goto doblot_exit_;
    }

    lanczos.lut = NULL;

    /* Some interpolation functions need some pre-calculated state */
    if (p->interpolation == interp_lanczos3 || p->interpolation == interp_lanczos5) {
        order = (p->interpolation == interp_lanczos3) ? 3 : 5;
        nlut = (size_t) ceil((double) order / lut_delta) + 1;
        if ((lanczos.lut = (double *) malloc(nlut * sizeof(double))) == NULL) {
            driz_error_set_message(p->error, "Out of memory");
            goto doblot_exit_;
        }

        create_lanczos_lut(order, nlut, lut_delta, lanczos.lut);

        lanczos.nbox = (integer_t) order;
        lanczos.nlut = nlut;
        lanczos.space = lut_delta;

        state = &lanczos;

    } /* Otherwise state is NULL */

    /* In the WCS case, we can't use the scale to calculate the Jacobian,
       so we need to do it.

       Note that we use the center of the image, rather than the reference pixel
       as the reference here.

       This is taken from dobox, except for the inversion of the image order.

       This section applies in WBLOT mode and now contains the addition
       correction to separate the distortion-induced scale change.
    */

    v = 1.0;

    for (j = 0; j < osize[1]; ++j) {
        /* Loop through the output positions and do the interpolation */
        for (i = 0; i < osize[0]; ++i) {
            if (oob_pixel(p->pixmap, i, j)) {
                driz_error_format_message(p->error, "OOB in pixmap[%d,%d]", i, j);
                return 1;
            }

            xo = get_pixmap(p->pixmap, i, j)[0];
            yo = get_pixmap(p->pixmap, i, j)[1];

            /* Check it is on the input image; NaN means no mapping is
               defined for this output pixel and it is treated as missing */
            if (!npy_isnan(xo) && !npy_isnan(yo) && xo >= 0.0 && xo < (double) isize[0] &&
                yo >= 0.0 && yo < (double) isize[1]) {
                /* Check for look-up-table interpolation */
                if (interpolate(state, &p->data_image, xo, yo, &v, p->error)) {
                    goto doblot_exit_;
                }

                if (oob_pixel(p->output_data, i, j)) {
                    driz_error_format_message(p->error, "OOB in output_data[%d,%d]", i, j);
                    return 1;
                } else {
                    image_set(&p->output_data_image, i, j, v * s);
                }

            } else {
                /* If there is nothing for us then set the output to missing C
                   value flag */
                image_set(&p->output_data_image, i, j, p->fill_value);
                p->nmiss++;
            }
        }
    }

doblot_exit_:
    driz_log_message("ending doblot");
    if (lanczos.lut) {
        free(lanczos.lut);
    }

    return driz_error_is_set(p->error);
}
