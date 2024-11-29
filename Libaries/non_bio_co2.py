# Import BW25 packages
import bw2data as bd
import brightway2 as bw 


def remove_bio_co2_recipe():
    all_methods = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H)' in str(m) and 'no LT' not in str(m)] # Midpoint

    # Obtaining the endpoint categories and ignoring land transformation
    endpoint = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H)' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

    method_name_new_mid= all_methods[0][0] + ' Runes edition'
    method_name_new_end = endpoint[0][0] + ' Runes edition'

    #Checking if OG is present in endpoint and then deleting it
    if 'OG' in method_name_new_end:
            method_name_new_end = method_name_new_end.replace(f'OG ', '')

    # Checking if the method exist
    if method_name_new_mid not in [m[0] for m in list(bd.methods)] or method_name_new_end not in [m[0] for m in list(bd.methods)]:
        # Combining mid- and endpoint method into one method
        for method in endpoint:
            all_methods.append(method)

        # Dictionary to store new method names
        new_methods = {}
        check = {}

        # For loop for setting bio-genic CO2 factor to 0
        for metod in all_methods:
            recipe_no_bio_CO2 = []  # Temporary storage for filtered CFs
            # Ensuring that the method only will be handled once
            if metod[1] not in check.keys():
                check[metod[1]] = None
                method = bw.Method(metod)
                cf_data = method.load()
                # filtering the CFs
                for cf_name, cf_value in cf_data:
                    flow_object = bw.get_activity(cf_name)
                    flow_name = flow_object['name'] if flow_object else "Unknown Flow"
                    if 'non-fossil' not in flow_name:
                        recipe_no_bio_CO2.append((cf_name, cf_value))
                    else:
                        recipe_no_bio_CO2.append((cf_name, 0))

                # registering the new method for later use
                new_metod = (metod[0] + ' Runes edition', metod[1], metod[2])
                new_method_key = new_metod
                new_method = bw.Method(new_method_key)
                new_method.register()
                new_method.write(recipe_no_bio_CO2)

                # Step 6: Store the new method
                new_methods[metod] = new_method_key
                print(f"New method created: {new_method_key} with {len(recipe_no_bio_CO2)} CFs")

    


            