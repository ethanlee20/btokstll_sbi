from pathlib import Path

from numpy import sqrt, zeros, sign, pi, arccos
from pandas import DataFrame, Series, concat, read_parquet
from uproot import open
from tqdm import tqdm

from .dict import flatten_dict
from .json_ import load_json


def error_on_length_mismatch(a: DataFrame|Series, b: DataFrame|Series):
    length_a = a.shape[0]
    length_b = b.shape[0]
    if not (length_a == length_b):
        raise ValueError(f"Lengths must be equal. Got {length_a} and {length_b}.")


def error_on_index_mismatch(a:DataFrame|Series, b:Dataframe|Series):
    if not (a.index.equals(b.index)):
        raise ValueError("Indices must be equal.")


def square_matrix_transform(matrix:DataFrame, vector:DataFrame) -> DataFrame:
    """
    Multiply a dataframe of vectors by a dataframe of square matrices.
    """

    error_on_length_mismatch(matrix, vector)
    error_on_index_mismatch(matrix, vector)
    if not (sqrt(matrix.shape[1]) == vector.shape[1]):
        raise ValueError("Matrix must be square.")

    vector_length = vector.shape[1]

    out = DataFrame(
        data=zeros(shape=vector.shape),
        index=vector.index,
        columns=vector.columns,
        dtype="float64",
    )

    for i in range(vector_length):
        for j in range(vector_length):
            out.iloc[:, i] += (
                matrix.iloc[:, vector_length * i + j]
                * vector.iloc[:, j]
            )

    return out


def dot_product(vector_1:DataFrame, vector_2:DataFrame) -> Series:
    """
    Compute the dot products of two vector dataframes.
    """

    error_on_length_mismatch(vector_1, vector_2)
    error_on_index_mismatch(vector_1, vector_2)
    if not (vector_1.shape[1] == vector_2.shape[1]):
        raise ValueError("Vector dimensions must match.")

    vector_length = vector_1.shape[1]

    out = Series(
        data=zeros(len(vector_1)),
        index=vector_1.index,
        dtype="float64",
    )

    for dimension in range(vector_length):
        out += (
            vector_1.iloc[:, dimension]
            * vector_2.iloc[:, dimension]
        )

    return out


def vector_magnitude(vector: DataFrame) -> Series:
    """
    Compute the magnitude of each vector in a vector dataframe.
    """
    out = sqrt(dot_product(vector, vector))
    return out


def cosine_angle(vector_1: DataFrame, vector_2: DataFrame) -> Series:
    """
    Find the cosine of the angle between vectors in vector dataframes.
    """
    out = dot_product(vector_1, vector_2) / (
        vector_magnitude(vector_1) * vector_magnitude(vector_2)
    )
    return out


def cross_product_3d(three_vector_1:DataFrame, three_vector_2:DataFrame) -> DataFrame:
    """
    Find the cross product of vectors from two 3-dimensional vector dataframes.
    """

    error_on_length_mismatch(three_vector_1, three_vector_2)
    error_on_index_mismatch(three_vector_1, three_vector_2)
    if not (three_vector_1.shape[1] == three_vector_2.shape[1] == 3):
        raise ValueError("Vectors must be 3-dimensional.")

    three_vector_1 = three_vector_1.copy()
    three_vector_2 = three_vector_2.copy()
    three_vector_1.columns = ["x", "y", "z"]
    three_vector_2.columns = ["x", "y", "z"]

    out = DataFrame(
        data=zeros(shape=three_vector_1.shape),
        index=three_vector_1.index,
        columns=three_vector_1.columns,
        dtype="float64",
    )
    out["x"] = (
        three_vector_1["y"] * three_vector_2["z"]
        - three_vector_1["z"] * three_vector_2["y"]
    )
    out["y"] = (
        three_vector_1["z"] * three_vector_2["x"]
        - three_vector_1["x"] * three_vector_2["z"]
    )
    out["z"] = (
        three_vector_1["x"] * three_vector_2["y"]
        - three_vector_1["y"] * three_vector_2["x"]
    )
    return out


def unit_normal(three_vector_1:DataFrame, three_vector_2:DataFrame) -> DataFrame:
    """
    For planes specified by two three-vector dataframes, calculate the unit normal vectors.
    """

    normal_vector = cross_product_3d(
        three_vector_1, three_vector_2
    )

    unit_normal_vector = normal_vector.divide(
        vector_magnitude(normal_vector), axis="index"
    )

    return unit_normal_vector


def to_four_momentum(input:DataFrame) -> DataFrame:
    """
    Create a labeled four-momentum dataframe.
    """
    if not input.shape[1] == 4:
        raise ValueError("Input must have four columns.")
    out = input.copy()
    out.columns = ["E", "px", "py", "pz"]
    return out


def to_three_momentum(input:DataFrame) -> DataFrame:
    """
    Create a labeled three-momentum dataframe.
    """
    if not input.shape[1] == 3:
        raise ValueError("Input must have three columns.")
    out = dataframe_with_three_columns.copy()
    out.columns = ["px", "py", "pz"]
    return out


def four_mom_to_three_mom(input:DataFrame) -> DataFrame:
    input = to_four_momentum(input)
    out = input.drop(columns="E")
    out = to_three_momentum(out)
    return out


def to_three_velocity(input:DataFrame) -> DataFrame:
    """
    Create a labeled three-velocity dataframe.
    """
    if not input.shape[1] == 3:
        raise ValueError("Input must have three columns.")
    out = input.copy()
    out.columns = ["vx", "vy", "vz"]
    return out


def calc_inv_mass_sq(four_momentum_1: DataFrame, four_momentum_2: DataFrame) -> DataFrame:
    """
    Calculate squared invariant masses for two particle systems.
    """

    four_momentum_1 = to_four_momentum(four_momentum_1)
    four_momentum_2 = to_four_momentum(four_momentum_2)

    four_momentum_sum = four_momentum_1 + four_momentum_2    
    three_momentum_sum = four_mom_to_three_mom(four_momentum_sum)
    
    three_momentum_sum_mag_sq = vector_magnitude(three_momentum_sum) ** 2
    out = four_momentum_sum["E"] ** 2 - three_momentum_sum_mag_sq
    return out


def three_vel_from_four_mom(four_momentum: DataFrame) -> DataFrame:
    """
    Compute a three-velocity dataframe from a four-momentum dataframe.
    """
    four_momentum = to_four_momentum(four_momentum)
    three_momentum = four_mom_to_three_mom(four_momentum)
    out = three_momentum.multiply(1 / four_momentum["E"], axis=0)
    return out


def calc_lorentz_factor(three_velocity: DataFrame) -> Series:
    """
    Calculate a Lorentz factor series.
    """
    three_velocity = to_three_velocity(three_velocity)
    three_velocity_mag = vector_magnitude(three_velocity)
    out = 1 / sqrt(1 - three_velocity_mag**2)
    return out


def calc_lorentz_boost_matrix(three_velocity: DataFrame) -> DataFrame:
    """
    Calculate a Lorentz boost matrix dataframe.
    """

    three_velocity = to_three_velocity(three_velocity)
    three_velocity_mag = vector_magnitude(three_velocity)
    lorentz_factor = calc_lorentz_factor(three_velocity)

    out = DataFrame(
        data=zeros(shape=(three_velocity.shape[0], 16)),
        index=three_velocity.index,
        columns=[
            "b00",
            "b01",
            "b02",
            "b03",
            "b10",
            "b11",
            "b12",
            "b13",
            "b20",
            "b21",
            "b22",
            "b23",
            "b30",
            "b31",
            "b32",
            "b33",
        ],
    )
    out["b00"] = lorentz_factor
    out["b01"] = (
        -lorentz_factor * three_velocity["vx"]
    )
    out["b02"] = (
        -lorentz_factor * three_velocity["vy"]
    )
    out["b03"] = (
        -lorentz_factor * three_velocity["vz"]
    )
    out["b10"] = (
        -lorentz_factor * three_velocity["vx"]
    )
    out["b11"] = (
        1
        + (lorentz_factor - 1)
        * three_velocity["vx"] ** 2
        / three_velocity_mag**2
    )
    out["b12"] = (
        (lorentz_factor - 1)
        * three_velocity["vx"]
        * three_velocity["vy"]
        / three_velocity_mag**2
    )
    out["b13"] = (
        (lorentz_factor - 1)
        * three_velocity["vx"]
        * three_velocity["vz"]
        / three_velocity_mag**2
    )
    out["b20"] = (
        -lorentz_factor * three_velocity["vy"]
    )
    out["b21"] = (
        (lorentz_factor - 1)
        * three_velocity["vy"]
        * three_velocity["vx"]
        / three_velocity_mag**2
    )
    out["b22"] = (
        1
        + (lorentz_factor - 1)
        * three_velocity["vy"] ** 2
        / three_velocity_mag**2
    )
    out["b23"] = (
        (lorentz_factor - 1)
        * three_velocity["vy"]
        * three_velocity["vz"]
        / three_velocity_mag**2
    )
    out["b30"] = (
        -lorentz_factor * three_velocity["vz"]
    )
    out["b31"] = (
        (lorentz_factor - 1)
        * three_velocity["vz"]
        * three_velocity["vx"]
        / three_velocity_mag**2
    )
    out["b32"] = (
        (lorentz_factor - 1)
        * three_velocity["vz"]
        * three_velocity["vy"]
        / three_velocity_mag**2
    )
    out["b33"] = (
        1
        + (lorentz_factor - 1)
        * three_velocity["vz"] ** 2
        / three_velocity_mag**2
    )
    return out


def boost(reference_four_momentum: DataFrame, four_vector: DataFrame) -> DataFrame:
    """
    Lorentz boost a dataframe of four-vectors to a reference four momentum dataframe.
    """
    reference_three_velocity = three_vel_from_four_mom(
        reference_four_momentum
    )
    boost_matrix = calc_lorentz_boost_matrix(
        reference_three_velocity
    )
    out = square_matrix_transform(
        boost_matrix, four_vector
    )
    return out


def calc_cos_theta_lepton(
    pos_lepton_four_mom: DataFrame,
    neg_lepton_four_mom: DataFrame,
    b_meson_four_mom: DataFrame,
) -> Series:
    """
    Find the cosine of the lepton helicity angle for B -> K* l+ l-.
    """
    pos_lepton_four_mom = to_four_momentum(pos_lepton_four_mom)
    neg_lepton_four_mom = to_four_momentum(neg_lepton_four_mom)
    dilepton_four_mom = pos_lepton_four_mom + neg_lepton_four_mom

    pos_lepton_four_mom_dilepton_frame = boost(
        reference_four_momentum=dilepton_four_mom,
        four_vector=pos_lepton_four_mom,
    )
    pos_lepton_three_mom_dilepton_frame = four_mom_to_three_mom(pos_lepton_four_mom_dilepton_frame)

    dilepton_four_mom_b_frame = boost(
        reference_four_momentum=b_meson_four_mom,
        four_vector=dilepton_four_mom,
    )
    dilepton_three_mom_b_frame = four_mom_to_three_mom(dilepton_four_mom_b_frame)

    out = cosine_angle(
        dilepton_three_mom_b_frame,
        pos_lepton_three_mom_dilepton_frame,
    )
    return out


def calc_cos_theta_k(
    k_four_mom: DataFrame,
    k_star_four_mom: DataFrame,
    b_meson_four_mom: DataFrame,
) -> Series:
    """
    Find the cosine of the K* helicity angle for B -> K* l+ l-.
    """
    k_four_mom_k_star_frame = boost(
        reference_four_momentum=k_star_four_mom,
        four_vector=k_four_mom,
    )
    k_three_mom_k_star_frame = four_mom_to_three_mom(k_four_mom_k_star_frame)

    k_star_four_mom_b_frame = boost(
        reference_four_momentum=b_meson_four_mom,
        four_vector=k_star_four_mom,
    )
    k_star_three_mom_b_frame = four_mom_to_three_mom(k_star_four_mom_b_frame)

    out = cosine_angle(
        vector_1=k_star_three_mom_b_frame,
        vector_2=k_three_mom_k_star_frame,
    )
    return out


def calc_unit_norm_k_star_k_plane(
    b_meson_four_mom : DataFrame,
    k_star_four_mom: DataFrame,
    k_four_mom: DataFrame,
) -> DataFrame:
    """
    Calculate the unit normal vector to the plane made by the directions of the K* and K in B -> K* l+ l-.
    """

    k_four_mom_k_star_frame = boost(
        reference_four_momentum=k_star_four_mom,
        four_vector=k_four_mom,
    )
    k_three_mom_k_star_frame = four_mom_to_three_mom(k_four_mom_k_star_frame)

    k_star_four_mom_b_frame = boost(
        reference_four_momentum=b_meson_four_mom,
        four_vector=k_star_four_mom,
    )
    k_star_three_mom_b_frame = four_mom_to_three_mom(k_star_four_mom_b_frame)

    out = unit_normal(
        three_vector_1=k_three_mom_k_star_frame,
        three_vector_2=k_star_three_mom_b_frame,
    )
    return out


def calc_unit_norm_dilepton_pos_lepton_plane(
    b_meson_four_mom: DataFrame,
    pos_lepton_four_mom: DataFrame,
    neg_lepton_four_mom: DataFrame,
) -> Series:
    """
    Find the unit normal to the plane made by the direction vectors of the dilepton system and the positively charged lepton in B -> K* l+ l-.
    """
    pos_lepton_four_mom = to_four_momentum(pos_lepton_four_mom)
    neg_lepton_four_mom = to_four_momentum(neg_lepton_four_mom)
    dilepton_four_mom = pos_lepton_four_mom + neg_lepton_four_mom

    pos_lepton_four_mom_dilepton_frame = boost(
        reference_four_momentum=dilepton_four_mom,
        four_vector=pos_lepton_four_mom,
    )
    pos_lepton_three_mom_dilepton_frame = four_mom_to_three_mom(pos_lepton_four_mom_dilepton_frame)

    dilepton_four_mom_b_frame = boost(
        reference_four_momentum=b_meson_four_mom, 
        four_vector=dilepton_four_mom
    )
    dilepton_three_mom_b_frame = four_mom_to_three_mom(dilepton_four_mom_b_frame)

    out = unit_normal(
        three_vector_1=pos_lepton_three_mom_dilepton_frame,
        three_vector_2=dilepton_three_mom_b_frame,
    )
    return out


def calc_cos_chi(
    b_meson_four_mom: DataFrame,
    k_four_mom: DataFrame,
    k_star_four_mom: DataFrame,
    pos_lepton_four_mom: DataFrame,
    neg_lepton_four_mom: DataFrame,
) -> Series:
    """
    Calculate the cosine of the decay angle chi in B -> K* l+ l-.

    Chi is the angle between the K* K decay plane and the dilepton l+ decay plane.
    """

    unit_norm_k_star_k_plane = (
        calc_unit_norm_k_star_k_plane(
            b_meson_four_mom=b_meson_four_mom,
            k_star_four_mom=k_star_four_mom,
            k_four_mom=k_four_mom,
        )
    )
    unit_norm_dilepton_pos_lepton_plane = calc_unit_norm_dilepton_pos_lepton_plane(
        b_meson_four_mom=b_meson_four_mom,
        pos_lepton_four_mom=pos_lepton_four_mom,
        neg_lepton_four_mom=neg_lepton_four_mom,
    )

    out = dot_product(
        vector_1=unit_norm_k_star_k_plane,
        vector_2=unit_norm_dilepton_pos_lepton_plane,
    )
    return out


def calc_sign_chi(
    b_meson_four_mom: DataFrame,
    k_star_four_mom: DataFrame,
    k_four_mom: DataFrame,
    pos_lepton_four_mom: DataFrame,
    neg_lepton_four_mom: DataFrame,
) -> ndarray:
    """
    Calculate the sign of angle chi.
    """
    unit_norm_k_star_k_plane = calc_unit_norm_k_star_k_plane(
        b_meson_four_mom=b_meson_four_mom,
        k_star_four_mom=k_star_four_mom,
        k_four_mom=k_four_mom,
    )
    unit_norm_dilepton_pos_lepton_plane = calc_unit_norm_dilepton_pos_lepton_plane(
        b_meson_four_mom=b_meson_four_mom,
        pos_lepton_four_mom=pos_lepton_four_mom,
        neg_lepton_four_mom=neg_lepton_four_mom,
    )
    cross_prod_norms = cross_product_3d(
        three_vector_1=unit_norm_dilepton_pos_lepton_plane,
        three_vector_2=unit_norm_k_star_k_plane,
    )

    k_star_four_mom_b_frame = boost(
        reference_four_momentum=b_meson_four_mom,
        four_vector=k_star_four_mom,
    )
    k_star_three_mom_b_frame = four_mom_to_three_mom(k_star_four_mom_b_frame)

    dot_prod_cross_prod_k_star_three_mom = dot_product(
        vector_1=cross_prod_norms,
        vector_2=k_star_three_mom_b_frame,
    )

    out = sign(dot_prod_cross_prod_k_star_three_mom)
    return out


def calc_chi(
    b_meson_four_mom: DataFrame,
    k_four_mom: DataFrame,
    k_star_four_mom: DataFrame,
    pos_lepton_four_mom: DataFrame,
    neg_lepton_four_mom: DataFrame,
) -> Series:
    """
    Find the decay angle chi in B -> K* l+ l-.

    Chi is the angle between the K* K decay plane and the dilepton l+ decay plane. It ranges from 0 to 2*pi.
    """

    cos_chi = calc_cos_chi(
        b_meson_four_mom=b_meson_four_mom,
        k_four_mom=k_four_mom,
        k_star_four_mom=k_star_four_mom,
        pos_lepton_four_mom=pos_lepton_four_mom,
        neg_lepton_four_mom=neg_lepton_four_mom,
    )

    sign_chi = calc_sign_chi(
        b_meson_four_mom=b_meson_four_mom,
        k_star_four_mom=k_star_four_mom,
        k_four_mom=k_four_mom,
        pos_lepton_four_mom=pos_lepton_four_mom,
        neg_lepton_four_mom=neg_lepton_four_mom,
    )

    out = sign_chi * arccos(cos_chi)
    convert_to_positive_angles = lambda x: x.where(x > 0, x + 2*pi)
    out = convert_to_positive_angles(out)
    return out


def calc_dif_inv_mass_k_pi_k_star(
    k_four_mom: DataFrame, pi_four_mom: DataFrame
) -> Series:
    """
    Calcualate the difference between the invariant mass of the K pi system and the K*'s invariant mass (PDG value).
    """
    inv_mass_k_star = 0.892
    inv_mass_k_pi = sqrt(calc_inv_mass_sq(
        four_momentum_1=k_four_mom,
        four_momentum_2=pi_four_mom
    ))
    out = inv_mass_k_pi - inv_mass_k_star
    return out


def calc_vars(input:DataFrame, lepton_flavor:str):
    """
    Calculate detector and generator level variables of B -> K* l+ l- decays.

    Variables include q^2, cosine theta l, cosine theta K, cosine chi, chi, and the
    difference between K pi invariant mass and K* PDG invariant mass.
    """

    if lepton_flavor not in ("mu", "e"):
        raise ValueError(f"Lepton flavor must be 'mu' or 'e'. Got: {lepton_flavor}")

    b_meson_four_mom = input[["E", "px", "py", "pz"]]
    b_meson_four_mom_mc = input[["mcE", "mcPX", "mcPY", "mcPZ"]]
    pos_lepton_four_mom = input[[
        f"{lepton_flavor}_p_E",
        f"{lepton_flavor}_p_px",
        f"{lepton_flavor}_p_py",
        f"{lepton_flavor}_p_pz",    
    ]]
    pos_lepton_four_mom_mc = input[[
        f"{lepton_flavor}_p_mcE",
        f"{lepton_flavor}_p_mcPX",
        f"{lepton_flavor}_p_mcPY",
        f"{lepton_flavor}_p_mcPZ",
    ]]
    neg_lepton_four_mom = input[[
        f"{lepton_flavor}_m_E",
        f"{lepton_flavor}_m_px",
        f"{lepton_flavor}_m_py",
        f"{lepton_flavor}_m_pz",
    ]]
    neg_lepton_four_mom_mc = input[[
        f"{lepton_flavor}_m_mcE",
        f"{lepton_flavor}_m_mcPX",
        f"{lepton_flavor}_m_mcPY",
        f"{lepton_flavor}_m_mcPZ",
    ]]
    k_four_mom = input[["K_p_E", "K_p_px", "K_p_py", "K_p_pz"]]
    k_four_mom_mc = input[["K_p_mcE", "K_p_mcPX", "K_p_mcPY", "K_p_mcPZ"]]
    pi_four_mom = input[["pi_m_E", "pi_m_px", "pi_m_py", "pi_m_pz"]]
    pi_four_mom_mc = input[["pi_m_mcE", "pi_m_mcPX", "pi_m_mcPY", "pi_m_mcPZ"]]
    k_star_four_mom = input[["KST0_E", "KST0_px", "KST0_py", "KST0_pz"]]
    k_star_four_mom_mc = input[["KST0_mcE", "KST0_mcPX", "KST0_mcPY", "KST0_mcPZ"]]

    out = input.copy()

    out["q_sq"] = calc_inv_mass_sq(
        pos_lepton_four_mom,
        neg_lepton_four_mom,
    )
    out["q_sq_mc"] = calc_inv_mass_sq(
        pos_lepton_four_mom_mc,
        neg_lepton_four_mom_mc,
    )
    out[f"cos_theta_lepton"] = calc_cos_theta_lepton(
        pos_lepton_four_mom=pos_lepton_four_mom,
        neg_lepton_four_mom=neg_lepton_four_mom,
        b_meson_four_mom=b_meson_four_mom,
    )
    out["cos_theta_lepton_mc"] = calc_cos_theta_lepton(
        pos_lepton_four_mom=pos_lepton_four_mom_mc,
        neg_lepton_four_mom=neg_lepton_four_mom_mc,
        b_meson_four_mom=b_meson_four_mom_mc,
    )
    out["cos_theta_k"] = calc_cos_theta_k(
        k_four_mom=k_four_mom,
        k_star_four_mom=k_star_four_mom,
        b_meson_four_mom=b_meson_four_mom,
    )
    out["cos_theta_k_mc"] = calc_cos_theta_k(
        k_four_mom=k_four_mom_mc,
        k_star_four_mom=k_star_four_mom_mc,
        b_meson_four_mom=b_meson_four_mom_mc,
    )
    out["chi"] = calc_chi(
        b_meson_four_mom=b_meson_four_mom,
        k_four_mom=k_four_mom,
        k_star_four_mom=k_star_four_mom,
        pos_lepton_four_mom=pos_lepton_four_mom,
        neg_lepton_four_mom=neg_lepton_four_mom,
    )
    out["chi_mc"] = calc_chi(
        b_meson_four_mom=b_meson_four_mom_mc,
        k_four_mom=k_four_mom_mc,
        k_star_four_mom=k_star_four_mom_mc,
        pos_lepton_four_mom=pos_lepton_four_mom_mc,
        neg_lepton_four_mom=neg_lepton_four_mom_mc,
    )
    out["dif_inv_mass_k_pi_k_star"] = (
        calc_dif_inv_mass_k_pi_k_star(
            k_four_mom=k_four_mom,
            pi_four_mom=pi_four_mom,
        )
    )
    out["dif_inv_mass_k_pi_k_star_mc"] = (
        calc_dif_inv_mass_k_pi_k_star(
            k_four_mom=k_four_mom_mc,
            pi_four_mom=pi_four_mom_mc,
        )
    )
    return out