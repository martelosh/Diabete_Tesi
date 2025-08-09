def build_dynamic_system_prompt(result_class, prob):
    if result_class == 0 and prob >= 0.80:
        mood = "Ottima notizia: al momento il profilo è rassicurante."
        call = "Per stare ancora più tranquilla/o, vuoi prenotare una visita di controllo?"
    elif result_class == 0:
        mood = "Al momento non risultano segnali di diabete."
        call = "Vuoi prenotare comunque un controllo preventivo?"
    elif result_class == 1:
        mood = "Sono emersi segnali compatibili con pre-diabete."
        call = "È consigliata una visita: vuoi prenotare?"
    else:  # result_class == 2
        mood = "Il profilo indica un rischio compatibile con diabete."
        call = "È strettamente consigliata una visita quanto prima: vuoi prenotare?"

    base = (
        "Sei un assistente sanitario gentile e concreto. "
        "Se l’utente dice SÌ alla prenotazione, chiedi il *comune* "
        "e proponi i recapiti disponibili. "
        "Mantieni un tono rassicurante e informativo."
    )
    return f"{mood} {call}\n{base}"
