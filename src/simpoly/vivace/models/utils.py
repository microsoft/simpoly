import logging
from typing import Union

from e3nn import o3

LOG = logging.getLogger(__name__)


def tp_path_exists(
    irreps_in1: Union[str, o3.Irreps],
    irreps_in2: Union[str, o3.Irreps],
    ir_out: Union[str, o3.Irreps],
) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def build_irreps(
    input_irreps: o3.Irreps,
    hidden_irreps: o3.Irreps,
    final_irreps: o3.Irreps,
    n_layers: int,
    nonscalars_include_parity: bool,
    use_all_tp_paths: bool = False,
) -> tuple[list[o3.Irreps], list[o3.Irreps]]:
    """
    input_irreps: edge feaure irreps. which is usually 0e + 1o + ... + lmax parity
    hidden_irreps: irreps of the initial local environment, which is usually exactly the same as above
    """
    if use_all_tp_paths:
        LOG.warning(
            "Using all tensor product paths, this may lead to a large memory footprint. "
            "The final irreps is ignored, since all paths are used"
        )
    # - begin irreps -
    # start to build up the irreps for the iterated TPs
    tps_irreps = [input_irreps]

    for layer_idx in range(n_layers):
        if layer_idx == 0:
            # Add parity irreps
            acceptable_ir_out = []
            for mul, ir in hidden_irreps:
                if nonscalars_include_parity:
                    # add both parity options
                    acceptable_ir_out.append((1, (ir.l, 1)))
                    acceptable_ir_out.append((1, (ir.l, -1)))
                else:
                    # add only the parity option seen in the inputs
                    acceptable_ir_out.append((1, ir))

            acceptable_ir_out = o3.Irreps(acceptable_ir_out)
        else:
            # does nothing
            pass

        if (layer_idx == n_layers - 1) and not use_all_tp_paths:
            # ^ means we're doing the last layer
            # for energy prediction, no more TPs follow this, so only need scalars
            acceptable_ir_out = final_irreps

        # Make sure that there exists the path that leads to chosen out_irreps
        ir_out = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in acceptable_ir_out
                if tp_path_exists(input_irreps, hidden_irreps, ir)
            ]
        )
        # the argument to the next tensor product is the output of this one
        input_irreps = ir_out
        # now we have a list of irreps `tps_irreps` with the length of `n_layers + 1`
        tps_irreps.append(ir_out)
    # - end build irreps -

    if not use_all_tp_paths:
        # == Remove unneeded paths ==
        # Some paths may exist in intermediate TP, but then never be used in the output irreps. Here we drop them.
        out_irreps = tps_irreps[-1]
        new_tps_irreps = [out_irreps]
        for input_irreps in reversed(tps_irreps[:-1]):
            new_input_irreps = []
            for mul, arg_ir in input_irreps:
                for _, env_ir in hidden_irreps:
                    if any(i in out_irreps for i in arg_ir * env_ir):
                        # arg_ir is useful: arg_ir * env_ir has a path to something we want
                        new_input_irreps.append((mul, arg_ir))
                        # once its useful, we keep it no matter what
                        break
            new_input_irreps = o3.Irreps(new_input_irreps)
            new_tps_irreps.append(new_input_irreps)
            out_irreps = new_input_irreps

        assert len(new_tps_irreps) == len(tps_irreps)
        tps_irreps = list(reversed(new_tps_irreps))
        del new_tps_irreps
        assert tps_irreps[-1].lmax == 0

    tps_irreps_in = tps_irreps[:-1]
    tps_irreps_out = tps_irreps[1:]
    del tps_irreps

    logging.info("Irreps on the way:")
    for irr_in, irr_out in zip(tps_irreps_in, tps_irreps_out):
        logging.info(f"{irr_in} --> {irr_out}")

    return tps_irreps_in, tps_irreps_out
