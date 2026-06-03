import inspect

import gradio as gr

from src.inference import predict_listing


APP_CSS = """
.gradio-container {
    width: min(100%, 1120px) !important;
    margin: 0 auto !important;
    padding: clamp(12px, 3vw, 28px) !important;
}

#app-title h1 {
    font-size: clamp(1.65rem, 4vw, 2.35rem);
    line-height: 1.15;
    overflow-wrap: anywhere;
}

#app-title p {
    font-size: clamp(0.95rem, 2.5vw, 1.05rem);
    line-height: 1.45;
}

#primary-action {
    width: 100%;
    margin-top: 0.5rem;
}

#primary-action button {
    width: 100%;
    min-height: 48px;
}

#input-grid,
#summary-row,
#risk-row,
#terms-row {
    align-items: stretch;
}

#runtime-files textarea {
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
}

@media (max-width: 720px) {
    .gradio-container {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }

    #input-grid,
    #summary-row,
    #risk-row,
    #terms-row {
        flex-direction: column !important;
        gap: 10px !important;
    }

    #input-grid > div,
    #summary-row > div,
    #risk-row > div,
    #terms-row > div {
        width: 100% !important;
        min-width: 0 !important;
        flex: 1 1 100% !important;
    }

    textarea,
    input,
    select {
        font-size: 16px !important;
    }

    .examples,
    .table-wrap,
    .dataframe {
        overflow-x: auto !important;
    }
}

@media (max-width: 420px) {
    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    #app-title h1 {
        font-size: 1.45rem;
    }
}
"""


def _supports_parameter(callable_obj, parameter_name):
    return parameter_name in inspect.signature(callable_obj).parameters


def _blocks_kwargs():
    if _supports_parameter(gr.Blocks, "css") and not _supports_parameter(gr.Blocks.launch, "css"):
        return {"css": APP_CSS}
    return {}


def _launch_demo():
    if _supports_parameter(demo.launch, "css"):
        return demo.launch(css=APP_CSS)
    return demo.launch()


def _fmt_money(value, currency):
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):,.0f} {currency}"
    except (TypeError, ValueError):
        return str(value)


def advise(
    brand,
    model,
    year,
    mileage_km,
    fuel_type,
    transmission,
    body_type,
    engine_size_l,
    condition,
    title_status,
    drive,
    paint_color,
    location_country,
    location_region,
    currency,
    seller_price,
    seller_description,
):
    result = predict_listing(
        brand,
        model,
        year,
        mileage_km,
        fuel_type,
        transmission,
        body_type,
        engine_size_l,
        condition,
        title_status,
        drive,
        paint_color,
        location_country,
        location_region,
        currency,
        seller_price,
        seller_description,
    )

    if result.get("error"):
        return (
            result["error"],
            "Unavailable",
            result["price_status"],
            result["nlp_risk_label"],
            "Unavailable",
            result["risk_terms_found"],
            result["positive_terms_found"],
            "Unavailable",
            result["final_recommendation"],
            result["explanation"],
            result["model_confidence_note"],
            result["runtime_file_summary"],
        )

    seller_diff = (
        f"{_fmt_money(result['price_difference'], currency)} "
        f"({result['price_difference_percent']:.1f}% vs predicted)"
    )
    offer_range = (
        f"{_fmt_money(result['recommended_offer_low'], currency)} to "
        f"{_fmt_money(result['recommended_offer_high'], currency)}"
    )
    return (
        _fmt_money(result["predicted_price"], currency),
        seller_diff,
        result["price_status"],
        result["nlp_risk_label"],
        f"{result['nlp_risk_score']:.2f}",
        result["risk_terms_found"] or "None detected",
        result["positive_terms_found"] or "None detected",
        offer_range,
        result["final_recommendation"],
        result["explanation"],
        f"{result['model_used']}. {result['model_confidence_note']}",
        result["runtime_file_summary"],
    )


with gr.Blocks(title="Used-Car Fair Price and Listing Risk Advisor", **_blocks_kwargs()) as demo:
    gr.Markdown(
        "# Used-Car Fair Price and Listing Risk Advisor\n"
        "This app combines numeric ML price prediction with NLP seller-description risk analysis. "
        "It uses the predicted fair price and extracted text-risk features together to recommend a "
        "negotiation range and buyer-friendly next step.",
        elem_id="app-title",
    )

    with gr.Row(elem_id="input-grid"):
        with gr.Column(scale=1, min_width=280):
            brand = gr.Textbox(label="Brand", value="Toyota")
            model = gr.Textbox(label="Model", value="Corolla")
            year = gr.Number(label="Year", value=2018, precision=0)
            mileage_km = gr.Number(label="Mileage (km)", value=89000)
            fuel_type = gr.Dropdown(
                ["Petrol", "Diesel", "Hybrid", "Electric", "Other", "Unknown"],
                label="Fuel type",
                value="Petrol",
            )
            transmission = gr.Dropdown(
                ["Manual", "Automatic", "Other", "Unknown"],
                label="Transmission",
                value="Automatic",
            )
            body_type = gr.Dropdown(
                ["Sedan", "Hatchback", "SUV", "Wagon", "Coupe", "Van", "Pickup", "Other", "Unknown"],
                label="Body type",
                value="Sedan",
            )
            engine_size_l = gr.Number(label="Engine size (L)", value=1.8)
            condition = gr.Dropdown(
                ["excellent", "good", "fair", "salvage", "unknown"],
                label="Condition",
                value="excellent",
            )
        with gr.Column(scale=1, min_width=280):
            title_status = gr.Textbox(label="Title status", value="clean")
            drive = gr.Textbox(label="Drive", value="fwd")
            paint_color = gr.Textbox(label="Paint color", value="silver")
            location_country = gr.Textbox(label="Location country", value="USA")
            location_region = gr.Textbox(label="Location region", value="ca")
            currency = gr.Textbox(label="Currency", value="USD")
            seller_price = gr.Number(label="Seller price", value=14200)
            seller_description = gr.Textbox(
                label="Seller description",
                lines=6,
                value="One owner Toyota Corolla, full service history, recently serviced, new tires, clean title, no accident.",
            )

    run_button = gr.Button("Analyze listing", variant="primary", elem_id="primary-action")

    with gr.Row(elem_id="summary-row"):
        with gr.Column(scale=1, min_width=220):
            predicted_price = gr.Textbox(label="Predicted fair price")
        with gr.Column(scale=1, min_width=220):
            seller_difference = gr.Textbox(label="Seller price difference")
        with gr.Column(scale=1, min_width=220):
            price_status = gr.Textbox(label="Price status")
    with gr.Row(elem_id="risk-row"):
        with gr.Column(scale=1, min_width=220):
            nlp_risk = gr.Textbox(label="NLP risk level")
        with gr.Column(scale=1, min_width=220):
            risk_score = gr.Textbox(label="Risk score")
        with gr.Column(scale=1, min_width=220):
            offer_range = gr.Textbox(label="Recommended offer range")
    with gr.Row(elem_id="terms-row"):
        with gr.Column(scale=1, min_width=220):
            risk_terms = gr.Textbox(label="Risk terms found")
        with gr.Column(scale=1, min_width=220):
            positive_terms = gr.Textbox(label="Positive terms found")
    final_recommendation = gr.Textbox(label="Final recommendation")
    explanation = gr.Textbox(label="Buyer-friendly explanation", lines=4)
    model_note = gr.Textbox(label="Model/confidence note", lines=3)
    runtime_files = gr.Textbox(label="Runtime files used", lines=18, elem_id="runtime-files")

    inputs = [
        brand,
        model,
        year,
        mileage_km,
        fuel_type,
        transmission,
        body_type,
        engine_size_l,
        condition,
        title_status,
        drive,
        paint_color,
        location_country,
        location_region,
        currency,
        seller_price,
        seller_description,
    ]
    outputs = [
        predicted_price,
        seller_difference,
        price_status,
        nlp_risk,
        risk_score,
        risk_terms,
        positive_terms,
        offer_range,
        final_recommendation,
        explanation,
        model_note,
        runtime_files,
    ]
    run_button.click(advise, inputs=inputs, outputs=outputs)

    gr.Examples(
        examples=[
            [
                "Toyota",
                "Corolla",
                2018,
                89000,
                "Petrol",
                "Automatic",
                "Sedan",
                1.8,
                "excellent",
                "clean",
                "fwd",
                "silver",
                "USA",
                "ca",
                "USD",
                14200,
                "One owner Toyota Corolla, full service history, recently serviced, new tires, clean title, no accident.",
            ],
            [
                "Volkswagen",
                "Golf",
                2016,
                126000,
                "Diesel",
                "Manual",
                "Hatchback",
                2.0,
                "good",
                "clean",
                "fwd",
                "black",
                "USA",
                "wa",
                "USD",
                9800,
                "VW Golf diesel, minor scratches on bumper, small rust spot, no inspection available, service history, no accident.",
            ],
            [
                "BMW",
                "328i",
                2013,
                171000,
                "Petrol",
                "Automatic",
                "Sedan",
                2.0,
                "fair",
                "rebuilt",
                "rwd",
                "gray",
                "USA",
                "fl",
                "USD",
                9600,
                "BMW 328i sold as is, oil leak, check engine light, needs repair, mechanic special.",
            ],
        ],
        inputs=inputs,
        outputs=outputs,
        fn=advise,
        cache_examples=False,
    )


if __name__ == "__main__":
    _launch_demo()
