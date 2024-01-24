simple_email_pattern = r"\S+@\S+\.\S+"
normal_email_pattern = r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[" \
                       r"a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"

simple_phone_pattern = r"\\+?[1-9][0-9]{7,14}"
normal_phone_pattern = r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"

normal_url_pattern = r"(?:https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[" \
                     r"a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{" \
                     r"2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
normal_url_pattern_v2 = r"(?:https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(?:\.[a-zA-Z]{2," \
                        r"})(?:\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2," \
                        r"}|(?:(?:https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(?:\.[a-zA-Z]{2," \
                        r"})(?:\.[a-zA-Z]{2,})?)|(?:https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{" \
                        r"2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
