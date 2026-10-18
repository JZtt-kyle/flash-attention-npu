"""Independent softcap risks, paired dtypes; layouts remain the public step's concern."""


def cases(representation):
    specs = [
        dict(
            id="rounding-single",
            shape=(1, 1, 8, 33),
            m=32,
            n=32,
            splits=1,
            scale=0.0,
            window=(1, 0),
            pattern="rounding-ties",
        ),
        dict(
            id="rounding-split",
            shape=(1, 1, 8, 33),
            m=32,
            n=32,
            splits=2,
            scale=0.0,
            window=(1, 0),
            pattern="rounding-ties",
        ),
        dict(
            id="rounding-bypass",
            shape=(1, 1, 8, 33),
            m=32,
            n=32,
            splits=2,
            scale=0.0,
            window=(1, 0),
            pattern="rounding-ties",
            softcap=0,
        ),
        dict(id="large-cap-near-zero", shape=(1, 3, 8, 257), m=32, n=128, splits=2, softcap=1e8),
        dict(id="small-cap-saturation", shape=(1, 3, 8, 257), m=32, n=128, splits=1, softcap=1e-4),
        dict(id="bypass-none", shape=(1, 3, 8, 257), m=32, n=128, splits=1, softcap=0),
        dict(
            id="bypass-causal",
            shape=(1, 17, 8, 257),
            m=128,
            n=128,
            splits=2,
            causal=True,
            softcap=0,
        ),
        dict(
            id="bypass-window",
            shape=(1, 9, 8, 257),
            m=128,
            n=64,
            splits=4,
            window=(70, 2),
            softcap=0,
        ),
        dict(id="softcap-single", shape=(1, 3, 8, 257), m=32, n=128, splits=1),
        dict(
            id="softcap-causal",
            shape=(2, 17, 8, 257),
            m=128,
            n=128,
            splits=2,
            causal=True,
            lengths=[257, 7],
        ),
        dict(id="softcap-window", shape=(1, 17, 8, 513), m=128, n=128, splits=4, window=(256, 2)),
        dict(
            id="zero-scale-used",
            shape=(2, 9, 4, 65),
            m=32,
            n=32,
            splits=2,
            scale=0.0,
            window=(3, 0),
            lengths=[0, 65],
        ),
        dict(
            id="saturation",
            shape=(1, 3, 8, 1025),
            m=32,
            n=128,
            splits=2,
            scale=0.23,
            pattern="saturation",
        ),
        dict(
            id="component-cancellation",
            shape=(1, 3, 8, 257),
            m=32,
            n=128,
            splits=2,
            scale=0.23,
            pattern="cancellation",
        ),
        dict(
            id="negative-scale-cap30",
            shape=(1, 17, 8, 65),
            m=128,
            n=64,
            splits=1,
            scale=-0.23,
            softcap=30,
            window=(33, -1),
        ),
        dict(id="empty-fragments-rows", shape=(1, 33, 8, 3), m=128, n=32, splits=8, causal=True),
        dict(id="nonbrc-tail", shape=(1, 65, 8, 513), m=256, n=128, splits=1, window=(-1, 3)),
    ]
    if representation == "prefill":
        specs = [dict(s, shape=(*s["shape"], 2)) for s in specs]
        specs += [
            dict(
                id="compare-mask-tail32",
                shape=(1, 1, 2, 33, 2),
                m=512,
                n=32,
                splits=1,
                pattern="saturation",
            ),
            dict(
                id="compare-mask-tail480",
                shape=(1, 17, 2, 513, 2),
                m=512,
                n=480,
                splits=1,
                pattern="saturation",
            ),
            dict(id="wide-scratch", shape=(1, 1, 8, 8193, 1), m=32, n=8192, splits=2),
            dict(
                id="cost-boundary",
                shape=(1, 257, 8, 4097, 2),
                m=512,
                n=512,
                splits=8,
                window=(2047, 1),
            ),
        ]
    elif representation == "compressed":
        specs += [
            dict(
                id="compare-mask-tail32",
                shape=(1, 1, 1, 33),
                m=32,
                n=32,
                splits=1,
                pattern="saturation",
            ),
            dict(id="cost-boundary", shape=(1, 3, 128, 16384), m=256, n=512, splits=2),
        ]
    else:
        raise ValueError("Unknown MLA representation")
    return [dict(softcap=15, **s) if "softcap" not in s else s for s in specs]


REPRESENTATIVES = {
    "rounding-single",
    "rounding-split",
    "rounding-bypass",
    "zero-scale-used",
    "large-cap-near-zero",
    "small-cap-saturation",
    "softcap-single",
    "softcap-causal",
    "saturation",
    "component-cancellation",
    "wide-scratch",
    "cost-boundary",
}
PERFORMANCE = {"bypass-none", "bypass-causal", "softcap-single", "softcap-window"}
