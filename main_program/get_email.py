import time
import imaplib
import email
from email.header import decode_header

# Přihlašovací údaje k e-mailovému účtu
imap_server = "imap.gmail.com"
username = r'matej.python.code@gmail.com'
password = r'ecfvvsdzjeykschq'

try:
    # Připojení k IMAP serveru
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(username, password)
    mail.select("inbox")


    # Převedení data na požadovaný formát "dd-MMM-yyyy"
    today_formatted_date = time.strftime("%d-%b-%Y", time.localtime(time.time() - (1.5 * 24 * 60 * 60)))

    search_date = today_formatted_date  # "23-Aug-2023"
    look_for_subject = "Your Subject Here"
    look_for_mail = r'matejporubsky@gmail.com'
    search_word = "obsah"
    tag = 'important'

    """
    SINCE:          Hledá e-maily odeslané od zadaného data.
    BEFORE:         Hledá e-maily odeslané před zadaným datem.
    ON:             Hledá e-maily odeslané v zadaný den.
    
    AND
    OR
    NOT
    """

    filters = ['ALL',
               f'SUBJECT {look_for_subject}',
               f'SINCE {search_date}',
               f'FROM {look_for_mail}',
               f'TO {look_for_mail}',
               f'TEXT {search_word}',
               f'KEYWORD {tag}',
               f'(FROM "{look_for_mail}" SINCE "{search_date}")',
               f'OR SINCE "{search_date}" SUBJECT "{look_for_subject}"',
               f'NOT FROM "{look_for_mail}"']

    # Hledání e-mailů s určitým předmětem
    status, email_ids = mail.search(None, f'{filters[5]}')

    # Získání seznamu ID nalezených e-mailů
    email_id_list = email_ids[0].split()

    for email_id in email_id_list:
        # Načtení e-mailu podle ID
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        try:
            print("From:", msg.get("From").split("==?= ")[1])
        except IndexError:
            print("From:", msg.get("From"))

        # Dekódování předmětu
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8")
        print("Subject:", subject)

        print()

        # Obsah zprávy
        if msg.is_multipart():
            count = 0
            for part in msg.walk():
                if not count == 0:
                    continue
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" not in content_disposition:
                    body = part.get_payload(decode=True)
                    if body:
                        print(body.decode("utf-8"))
                        count += 1
        else:
            body = msg.get_payload(decode=True)
            if body:
                print(body.decode("utf-8"))

        # Zde můžete provést další operace na základě předmětu nebo obsahu e-mailu
        if subject == f"{look_for_subject}":
            print("E-mail with subject found:")
            print("Subject:", subject)
            print("From:", msg.get("From"))
            print("Content:")
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if "attachment" not in content_disposition:
                        body = part.get_payload(decode=True)
                        print(body.decode("utf-8"))
                else:
                    body = msg.get_payload(decode=True)
                    print(body.decode("utf-8"))

        print("=" * 80)

    # Odhlášení z e-mailového účtu
    mail.logout()

except imaplib.IMAP4.error as e:
    print(f"\tFail to connect to server\n\t\tDESCRIPTION:{e}")
