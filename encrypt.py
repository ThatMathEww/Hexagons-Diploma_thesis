from cryptography.fernet import Fernet

# Generujte náhodný klíč pro šifrování
key_ = Fernet.generate_key()

key = '2?gKUpT.QXD7hY*D&@jlRpSCV%w39GQ=!amG_WTI7fuwE;Ncb,okQ='
key = key.encode('utf-8')

# Vytvořte objekt Fernet s tímto klíčem
fernet = Fernet(key)

# Data k zašifrování (Unicode řetězec)
data = 'Toto jsou data, která budou zašifrována.'

# Převeďte Unicode řetězec na bytový řetězec v kódování UTF-8
data = data.encode('utf-8')


print("\nklíč:", key)

# Zašifrujte data
zasifrovana_data = fernet.encrypt(data)

print('\tZašifrovaná data:', zasifrovana_data)

# Pro dešifrování použijte stejný klíč
rozsifrovana_data = fernet.decrypt(zasifrovana_data)
# Převeďte bytový řetězec zpět na Unicode řetězec
rozsifrovana_data = rozsifrovana_data.decode('utf-8')

print('Rozšifrovaná data:', rozsifrovana_data)
