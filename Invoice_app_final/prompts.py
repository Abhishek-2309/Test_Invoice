kv_prompt = """
  You are given the markdown of an Invoice document with tables enclosed within '<table>' tags.
  Map all the fields in the below JSON schema to the closest value associated with it that is present in the markdown input.
  Note:
  For the Json key of "Main_Table", identify the main line table from the available tables containing the line items of the Invoice document and only add all the items present in that table. Do Not include
  totals and others.
  Subtotal is the total amount of line items before taxes, 'Taxes' refers to total value of taxes, Total Amount Due is the Subtotal + Taxes
  Return it as follows:
  ~~~json
  {{
    "Header": {{
      "Invoice Number": "...",
      "Invoice Date": "...",
      "Seller's Information": {{
        "Company Name": "...",
        "Address": "...",
        "Contact": "...",
        "GSTIN": "..."
      }},
      "Buyer's Information": {{
        "Company Name": "...",
        "Address": "...",
        "Contact": "...",
        "GSTIN": "..."
      }}
    }},
    "Main_Table":
          {{
        "items": [
          {{ "<column1>": "value", ... }},
          ...
        ]
    }},
    "Payment Terms": {{
      "Bank_details": {{
        "Bank Name": "...",
        "IFSC_code": "...",
        "bank_account_no": "..."
      }},
      "Payment Due Date": "...",
      "Payment Methods": "..."
    }},
    "Summary": {{
      "Subtotal": "...",
      "Taxes": "...",
      "Discounts": "...",
      "Total Amount Due": "..."
    }},
    "Other Important Sections": {{
      "Terms and conditions": "...",
      "Notes/Comments": "...",
      "Signature": "..."
    }}
  }}
  ~~~
  Extract only from document contents, If relevant values are not found, leave empty.
  Markdown:
  {doc_body}
  """
