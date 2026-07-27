def calc_num_events(
    int_lum_inv_ab: float = 0.8,
    B_anti_B_cross_sec_nb: float = 1.1,
    B0_anti_B0_branch_frac: float = 0.483,
    B0_to_K_star_mu_mu_branch_frac: float = 0.000001200,
    anti_B0_to_anti_K_star_mu_mu_branch_frac: float = 0.000001200,
    B0_to_K_star_e_e_branch_frac: float = 0.000001210,
    anti_B0_to_anti_K_star_e_e_branch_frac: float = 0.000001210,
    efficiency: float = 0.25,
):
    num_B_anti_B_events = int_lum_inv_ab * B_anti_B_cross_sec_nb * 1e-9 * 1e18

    num_B0_anti_B0_events = B0_anti_B0_branch_frac * num_B_anti_B_events

    num_K_star_mu_mu_events = B0_to_K_star_mu_mu_branch_frac * num_B0_anti_B0_events
    num_anti_K_star_mu_mu_events = (
        anti_B0_to_anti_K_star_mu_mu_branch_frac * num_B0_anti_B0_events
    )
    num_K_star_e_e_events = B0_to_K_star_e_e_branch_frac * num_B0_anti_B0_events
    num_anti_K_star_e_e_events = (
        anti_B0_to_anti_K_star_e_e_branch_frac * num_B0_anti_B0_events
    )

    num_events = {
        "K_star_mu_mu": num_K_star_mu_mu_events,
        "anti_K_star_mu_mu": num_anti_K_star_mu_mu_events,
        "K_star_e_e": num_K_star_e_e_events,
        "anti_K_star_e_e": num_anti_K_star_e_e_events,
    }

    num_events_after_efficiency = {
        decay: efficiency * num for decay, num in num_events.items()
    }

    return num_events_after_efficiency


if __name__ == "__main__":

    num = calc_num_events(int_lum_inv_ab=50)
    print(num)
