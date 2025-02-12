# Importing libraries
import bw2io as bi
import bw2data as bd
import pandas as pd

def import_excel_database_to_brightway(data, sheet):
    # Save the data to a temporary file that can be used by ExcelImporter
    temp_path = r'C:\Users\ruw\Desktop\RA\Single-use-vs-multi-use-in-health-care\Data\databases\temp.xlsx'
    data.to_excel(temp_path, index=False)
    
    # Use the temporary file with ExcelImporter
    imp = bi.ExcelImporter(temp_path)  # the path to your inventory excel file
    imp.apply_strategies()

    # Create the database if it hasn't been called from reload_database
    if reload_database.has_been_called == False:
        imp.write_database()

    # Loop through each database and match
    print(f"Matching database: {sheet}")
    imp.match_database(sheet, fields=('name', 'unit', 'location', 'reference product'))
    print(f"Unlinked items after matching {sheet}: {list(imp.unlinked)}")

    # Match without specifying a database
    imp.match_database(fields=('name', 'unit', 'location'))

    # Generate statistics and write results
    imp.statistics()
    imp.write_excel(only_unlinked=True)
    unlinked_items = list(imp.unlinked)
    imp.write_database()

    # Print unlinked items if needed
    print(unlinked_items)
    print(f'{data.columns[1]} is loaded into the database')
    import_excel_database_to_brightway.has_been_called = True

def reload_database(sheet_name, system_path):
    reload_database.has_been_called = True
    user_input = input('Do you want to reload some or all the databases? [y/n or a for all]')
    
    if user_input.lower() == 'y':
        # Reload specific databases
        for case, path in enumerate(system_path):    
            db_path = path 
            for db in sheet_name:
                data = pd.read_excel(db_path, sheet_name=db)
                user_input2 = input(f'Reload case{case+1}_{db}? [y/n]')
                if user_input2.lower() == 'y':
                    import_excel_database_to_brightway(data, db)

    elif user_input.lower() == 'a':
        # Reload all databases
        for case, path in enumerate(system_path):    
            db_path = path 
            for db in sheet_name:
                data = pd.read_excel(db_path, sheet_name=db)
                import_excel_database_to_brightway(data, db)

    elif user_input.lower() == 'n':
        print('You selected to not reload')

    else:
        print('Invalid argument, try again')
        reload_database(sheet_name, system_path)

def database_setup(ecoinevnt_paths, system_path, bw_project="Single Use vs Multi Use", sheet_names = ['ev391apos', 'ev391consq', 'ev391cutoff']):
    # Set the current Brightway project
    bd.projects.set_current(bw_project)

    # Check if biosphere database is already present
    if any("biosphere" in db for db in bd.databases):
        print('Biosphere is already present in the project.')
    else:
        bi.bw2setup()

    # Check if Ecoinvent databases are already present
    if 'ev391cutoff' in bd.databases and 'ev391consq' in bd.databases and 'ev391apos' in bd.databases:
        print('Ecoinvent 3.9.1 is already present in the project.')
    else:
        # Import APOS database
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391apos'], db_name='ev391apos')
        ei.apply_strategies()
        ei.statistics()
        ei.write_database()

        # Import Consequential database
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391consq'], db_name='ev391consq')
        ei.apply_strategies()
        ei.statistics()
        ei.write_database()

        # Import Cut-off database
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391cutoff'], db_name='ev391cutoff')
        ei.apply_strategies()
        ei.statistics()
        ei.write_database()

    # Import Excel databases
    for path in system_path:    
        for sheet in sheet_names:
            # Read the Excel file
            data = pd.read_excel(path, sheet_name=sheet)
            
            import_excel_database_to_brightway.has_been_called = False  
            if data.columns[1] not in bd.databases:
                import_excel_database_to_brightway(data, sheet)
    
    # Reload databases if needed
    if import_excel_database_to_brightway.has_been_called is False:
        reload_database(sheet_names, system_path)