def send_message(send_sms=False, send_mail=False, message_sms="", message_mail="", name="", phone_number=r"737192215",
                 mail_address=r'matejporubsky@gmail.com', formatted_time=None):
    import smtplib

    # Přihlašovací údaje k účtu Gmail
    smtp_server = r'smtp.gmail.com'
    smtp_port = 587
    email_address = r'matej.python.code@gmail.com'
    password = r'ecfvvsdzjeykschq'  # r'#Python.M@tej951'

    try:
        if send_sms:
            if len(phone_number) == 9:
                if len(message_sms) > 26:
                    print(f"\n\tZpráva je příliž dlouhá\n\t\tMaximum znaků: 26\t[použito: {len(message_sms)}]")

                # Údaje pro odesílání SMS na O2
                # mobile_number = ['737192215', '777015903']
                o2_sms_gateway = f'+420{phone_number}@sms.cz.o2.com'
                # t_mobile_sms_gateway = f'{mobile_number[1]}@sms.t-mobile.cz'

                # Obsah zprávy
                # message = 'Message from program.'  # WARNING MAX length 26

                # Vytvoření e-mailové zprávy
                email_message = f'Subject: SMS\n\n{message_sms}'

                # Odeslání e-mailu
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(email_address, password)
                    server.sendmail(email_address, o2_sms_gateway, email_message)

                print("\tSMS was successfully sent.")

            else:
                print(f"Wrong phone formate: '{phone_number}'\n\tLength of number is {len(phone_number)}, not 9.")

        if send_mail:
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.utils import formataddr, parseaddr

            def is_valid_email(address):
                # Zkontrolujte, zda je e-mailová adresa ve správném formátu
                _, address = parseaddr(address)
                return bool(address)

            # Zkontrolujte platnost e-mailové adresy
            if not is_valid_email(mail_address):
                print(f"Invalid e-mail address of the recipient.\n\t[{mail_address}]")
            else:
                # Adresa příjemce
                # emails = [r'matejma@seznam.cz', r'nahradnikO1@seznam.cz',
                #           r'matejporubsky@gmail.com', r'matejma38@gmail.com']
                # to_email = emails[-2]

                if formatted_time is None:
                    import time
                    formatted_time = time.strftime("%H-%M-%S_%d-%m-%Y", time.localtime(time.time()))

                # Vytvoření e-mailu
                message = MIMEMultipart()
                message["From"] = formataddr((f"Python_program:  {name}", email_address))
                # message['From'] = email_address
                message['To'] = mail_address  # to_email
                message['Subject'] = f'Python program is finished: _[{formatted_time}]_'

                # Tělo zprávy
                message.attach(MIMEText(message_mail, 'plain'))

                # Odeslání e-mailu
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()

                server.login(email_address, password)

                server.send_message(message)

                print("\tE-mail was successfully sent.")

    except smtplib.SMTPException as e:
        print(f"\tSending messages failed.\n\t\tDESCRIPTION:{e}")
